//! Persistence: an append-only, hash-chained journal of everything the coordinator measured or
//! decided, and SHA-256 envelopes for checkpoints (learned state, active settings).
//!
//! Journal: one canonical JSON record per line, `{seq, prev, time_ms, identity, kind, body, hash}`
//! with `hash = sha256(record without hash)` and `prev` the previous record's hash. Opening verifies
//! the whole chain. A head file (`<journal>.head`: last sequence number and hash, written atomically
//! after each synced append) catches records cut from the end. A final line without its newline is
//! a torn write from a crash only if the head does not already count it: it is copied to
//! `<journal>.torn-<n>` and cut off (atomically), never silently dropped. Any other damage (a changed
//! byte, a missing, reordered or truncated record, a head ahead of the file) refuses the journal: the
//! caller keeps the reference and stops. Deleting the journal and its head together is not detected.
//!
//! Envelope: `{schema, identity, payload, sha256}`, written to a temporary file, synced and renamed,
//! so a crash leaves the old or the new file, never half of one. Reading checks schema, identity and
//! digest; a mismatch is reported so the caller can quarantine the file and start from the reference.

use crate::json::{self, Value};
use crate::sha256;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

pub const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

#[derive(Debug)]
pub struct Journal {
    path: PathBuf,
    pub records: Vec<Value>,
    last_hash: String,
    pub torn_recovered: bool,
}

fn record_hash(record: &Value) -> String {
    let mut body = record.clone();
    if let Value::Obj(map) = &mut body {
        map.remove("hash");
    }
    sha256::hex_digest(body.dump().as_bytes())
}

impl Journal {
    pub fn open(path: &Path) -> Result<Journal, String> {
        let mut journal = Journal { path: path.to_path_buf(), records: Vec::new(), last_hash: GENESIS.into(), torn_recovered: false };
        let text = match fs::read(path) {
            Ok(bytes) => String::from_utf8(bytes).map_err(|_| "journal is not UTF-8".to_string())?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(journal),
            Err(e) => return Err(format!("journal unreadable: {e}")),
        };
        let head_path = PathBuf::from(format!("{}.head", path.display()));
        let head = match fs::read_to_string(&head_path) {
            Ok(t) => Some(json::parse(t.trim()).map_err(|e| format!("journal head unreadable: {e}"))?),
            Err(_) => None,
        };
        let head_seq = head.as_ref().and_then(|h| h.num("seq")).map_or(-1.0, |s| s);
        let mut body = text.as_str();
        if !body.is_empty() && !body.ends_with('\n') {
            let complete = body[..body.rfind('\n').map_or(0, |i| i + 1)].lines().count() as f64;
            if complete <= head_seq {
                return Err("journal ends in a damaged record that the head counts as written".into());
            }
            let cut = body.rfind('\n').map_or(0, |i| i + 1);
            let mut n = 0;
            let torn = loop {
                let candidate = PathBuf::from(format!("{}.torn-{n}", path.display()));
                if !candidate.exists() {
                    break candidate;
                }
                n += 1;
            };
            write_atomic(&torn, body[cut..].as_bytes()).map_err(|e| format!("cannot keep torn tail: {e}"))?;
            body = &body[..cut];
            write_atomic(path, body.as_bytes()).map_err(|e| format!("cannot cut torn tail: {e}"))?;
            journal.torn_recovered = true;
        }
        for (i, line) in body.lines().enumerate() {
            let record = json::parse(line).map_err(|e| format!("journal line {} unreadable: {e}", i + 1))?;
            if record.num("seq") != Some(i as f64) {
                return Err(format!("journal line {} has sequence {:?}", i + 1, record.num("seq")));
            }
            if record.str("prev") != Some(journal.last_hash.as_str()) {
                return Err(format!("journal line {} breaks the chain", i + 1));
            }
            let hash = record.str("hash").ok_or(format!("journal line {} has no hash", i + 1))?.to_string();
            if record_hash(&record) != hash {
                return Err(format!("journal line {} was altered", i + 1));
            }
            journal.last_hash = hash;
            journal.records.push(record);
        }
        if let Some(h) = head {
            let n = journal.records.len() as f64;
            let at = h.num("seq").unwrap_or(-1.0);
            // The head may lag by one (crash after the append, before the head); it may never lead.
            if at > n - 1.0 {
                return Err(format!("journal was truncated: head names record {at}, the file holds {n}"));
            }
            if at >= 0.0 && journal.records.get(at as usize).and_then(|r| r.str("hash")) != h.str("hash") {
                return Err("journal head does not match its record".into());
            }
        }
        Ok(journal)
    }

    pub fn append(&mut self, identity: &str, kind: &str, body: Value) -> Result<Value, String> {
        let mut record = Value::obj()
            .with("seq", self.records.len())
            .with("prev", self.last_hash.as_str())
            .with("time_ms", now_ms())
            .with("identity", identity)
            .with("kind", kind)
            .with("body", body);
        let hash = record_hash(&record);
        record.set("hash", hash.as_str());
        let mut file = OpenOptions::new().create(true).append(true).open(&self.path)
            .map_err(|e| format!("journal not writable: {e}"))?;
        file.write_all(format!("{}\n", record.dump()).as_bytes()).map_err(|e| format!("journal write failed: {e}"))?;
        file.sync_data().map_err(|e| format!("journal sync failed: {e}"))?;
        let head = Value::obj().with("seq", self.records.len()).with("hash", hash.as_str());
        write_atomic(&PathBuf::from(format!("{}.head", self.path.display())), format!("{}\n", head.dump()).as_bytes())?;
        self.last_hash = hash;
        self.records.push(record.clone());
        Ok(record)
    }

    pub fn of_kind<'a>(&'a self, kind: &'a str) -> impl Iterator<Item = &'a Value> + 'a {
        self.records.iter().filter(move |r| r.str("kind") == Some(kind))
    }
}

/// Atomic write: temporary file, sync, rename.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    {
        let mut f = File::create(&tmp).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
        f.write_all(bytes).map_err(|e| format!("write failed: {e}"))?;
        f.sync_all().map_err(|e| format!("sync failed: {e}"))?;
    }
    fs::rename(&tmp, path).map_err(|e| format!("rename failed: {e}"))
}

pub fn write_envelope(path: &Path, schema: &str, identity: &str, payload: Value) -> Result<(), String> {
    let digest = sha256::hex_digest(payload.dump().as_bytes());
    let envelope = Value::obj().with("schema", schema).with("identity", identity)
        .with("payload", payload).with("sha256", digest);
    write_atomic(path, format!("{}\n", envelope.dump()).as_bytes())
}

#[derive(Debug, PartialEq)]
pub enum EnvelopeError {
    Missing,
    Corrupt(String),
    ForeignIdentity(String),
}

pub fn read_envelope(path: &Path, schema: &str, identity: &str) -> Result<Value, EnvelopeError> {
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(EnvelopeError::Missing),
        Err(e) => return Err(EnvelopeError::Corrupt(format!("unreadable: {e}"))),
    };
    let v = json::parse(text.trim()).map_err(|e| EnvelopeError::Corrupt(format!("not JSON: {e}")))?;
    if v.str("schema") != Some(schema) {
        return Err(EnvelopeError::Corrupt(format!("schema {:?}, expected {schema}", v.str("schema"))));
    }
    let payload = v.get("payload").cloned().ok_or(EnvelopeError::Corrupt("no payload".into()))?;
    if v.str("sha256") != Some(sha256::hex_digest(payload.dump().as_bytes()).as_str()) {
        return Err(EnvelopeError::Corrupt("digest mismatch".into()));
    }
    match v.str("identity") {
        Some(i) if i == identity => Ok(payload),
        other => Err(EnvelopeError::ForeignIdentity(other.unwrap_or("").to_string())),
    }
}

/// Move a damaged or foreign file aside (kept for inspection), returning where it went.
pub fn quarantine(path: &Path) -> Option<PathBuf> {
    let target = PathBuf::from(format!("{}.quarantined-{}", path.display(), now_ms()));
    fs::rename(path, &target).ok().map(|_| target)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("imopt-store-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn journal_chain_survives_reopen_detects_tampering_and_recovers_a_torn_tail() {
        let d = dir("journal");
        let p = d.join("journal.jsonl");
        let mut j = Journal::open(&p).unwrap();
        j.append("id", "measure", Value::obj().with("x", 1.0)).unwrap();
        j.append("id", "decide", Value::obj().with("y", "z")).unwrap();
        let j2 = Journal::open(&p).unwrap();
        assert_eq!(j2.records.len(), 2);
        // a torn final write is kept aside and cut, the chain stays valid
        let mut f = OpenOptions::new().append(true).open(&p).unwrap();
        f.write_all(b"{\"seq\":2,\"partial").unwrap();
        let mut j3 = Journal::open(&p).unwrap();
        assert!(j3.torn_recovered && j3.records.len() == 2);
        assert!(d.join("journal.jsonl.torn-0").exists());
        j3.append("id", "measure", Value::obj()).unwrap();
        assert_eq!(Journal::open(&p).unwrap().records.len(), 3);
        let original: Vec<String> = fs::read_to_string(&p).unwrap().lines().map(String::from).collect();
        // one altered byte inside the file is refused
        let text = fs::read_to_string(&p).unwrap().replace("\"y\":\"z\"", "\"y\":\"Z\"");
        fs::write(&p, text).unwrap();
        assert!(Journal::open(&p).unwrap_err().contains("altered"));
        // a removed record breaks the chain
        let lines: Vec<String> = fs::read_to_string(&p).unwrap().lines().map(String::from).collect();
        fs::write(&p, format!("{}\n{}\n", lines[0], lines[2])).unwrap();
        assert!(Journal::open(&p).is_err());
        // records cut from the end are caught by the head
        fs::write(&p, format!("{}\n{}\n", original[0], original[1])).unwrap();
        assert!(Journal::open(&p).unwrap_err().contains("truncated"));
    }

    #[test]
    fn envelopes_check_schema_digest_and_identity() {
        let d = dir("envelope");
        let p = d.join("state.json");
        assert_eq!(read_envelope(&p, "s", "id"), Err(EnvelopeError::Missing));
        write_envelope(&p, "s", "id", Value::obj().with("q", 1.5)).unwrap();
        assert_eq!(read_envelope(&p, "s", "id").unwrap().num("q"), Some(1.5));
        assert!(matches!(read_envelope(&p, "s", "other"), Err(EnvelopeError::ForeignIdentity(_))));
        assert!(matches!(read_envelope(&p, "t", "id"), Err(EnvelopeError::Corrupt(_))));
        let text = fs::read_to_string(&p).unwrap().replace("1.5", "2.5");
        fs::write(&p, text).unwrap();
        assert!(matches!(read_envelope(&p, "s", "id"), Err(EnvelopeError::Corrupt(_))));
        let moved = quarantine(&p).unwrap();
        assert!(moved.exists() && !p.exists());
    }
}
