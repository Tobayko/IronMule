//! Adapter 2's workload: an own, reproducible CPU job. A seeded text-like corpus is split into
//! chunks; each chunk is generated and compressed with a small greedy LZ77 coder. The job's output
//! is the SHA-256 over every compressed chunk in index order, so it does not depend on how many
//! threads did the work: that is the correctness rule. The reversible application setting is the
//! thread count. The foreground `probe` is a latency-sensitive request stream on the same CPUs.

use crate::sha256::{self, Sha256};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const WORDS: usize = 512;
const MIN_MATCH: usize = 4;
const MAX_MATCH: usize = 258;
const WINDOW: usize = 1 << 16;
const HASH_BITS: u32 = 15;

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn vocabulary(seed: u64) -> Vec<Vec<u8>> {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    (0..WORDS).map(|_| {
        let len = 2 + (xorshift(&mut s) % 9) as usize;
        (0..len).map(|_| b'a' + (xorshift(&mut s) % 26) as u8).collect()
    }).collect()
}

/// Chunk `index` of the corpus for `seed`: Zipf-like word draws, so it compresses about 3:1.
pub fn chunk(seed: u64, index: usize, size: usize, vocab: &[Vec<u8>]) -> Vec<u8> {
    let mut s = seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ (index as u64 + 1).wrapping_mul(0x9E37_79B9);
    let mut out = Vec::with_capacity(size + 16);
    while out.len() < size {
        let r = xorshift(&mut s);
        // rank ~ 1 / u: low ranks dominate
        let u = ((r >> 11) as f64 / (1u64 << 53) as f64).max(1e-9);
        let rank = ((1.0 / u) as usize).min(WORDS) - 1;
        out.extend_from_slice(&vocab[rank]);
        out.push(if r % 13 == 0 { b'\n' } else { b' ' });
    }
    out.truncate(size);
    out
}

fn put_varint(out: &mut Vec<u8>, mut v: usize) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn get_varint(input: &[u8], at: &mut usize) -> Option<usize> {
    let (mut v, mut shift) = (0usize, 0);
    loop {
        let b = *input.get(*at)?;
        *at += 1;
        v |= ((b & 0x7f) as usize) << shift;
        if b < 0x80 {
            return Some(v);
        }
        shift += 7;
        if shift > 56 {
            return None;
        }
    }
}

/// Greedy LZ77: [literal count][literals][match length][offset] ..., final literal run.
pub fn compress(data: &[u8]) -> Vec<u8> {
    let mut table = vec![usize::MAX; 1 << HASH_BITS];
    let mut out = Vec::with_capacity(data.len() / 2);
    let (mut i, mut lit_start) = (0usize, 0usize);
    let hash = |p: usize| -> usize {
        let v = u32::from_le_bytes(data[p..p + 4].try_into().unwrap());
        (v.wrapping_mul(2_654_435_761) >> (32 - HASH_BITS)) as usize
    };
    while i + MIN_MATCH <= data.len() {
        let h = hash(i);
        let candidate = table[h];
        table[h] = i;
        if candidate != usize::MAX && i - candidate <= WINDOW && data[candidate..candidate + MIN_MATCH] == data[i..i + MIN_MATCH] {
            let mut len = MIN_MATCH;
            while i + len < data.len() && len < MAX_MATCH && data[candidate + len] == data[i + len] {
                len += 1;
            }
            put_varint(&mut out, i - lit_start);
            out.extend_from_slice(&data[lit_start..i]);
            put_varint(&mut out, len);
            put_varint(&mut out, i - candidate);
            for p in i + 1..(i + len).min(data.len().saturating_sub(MIN_MATCH - 1)) {
                table[hash(p)] = p;
            }
            i += len;
            lit_start = i;
        } else {
            i += 1;
        }
    }
    put_varint(&mut out, data.len() - lit_start);
    out.extend_from_slice(&data[lit_start..]);
    out
}

pub fn decompress(input: &[u8]) -> Option<Vec<u8>> {
    let (mut out, mut at) = (Vec::new(), 0usize);
    loop {
        let lits = get_varint(input, &mut at)?;
        out.extend_from_slice(input.get(at..at + lits)?);
        at += lits;
        if at == input.len() {
            return Some(out);
        }
        let len = get_varint(input, &mut at)?;
        let off = get_varint(input, &mut at)?;
        if off == 0 || off > out.len() || len < MIN_MATCH {
            return None;
        }
        let start = out.len() - off;
        for k in 0..len {
            let b = out[start + k];
            out.push(b);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JobSpec {
    pub seed: u64,
    pub chunks: usize,
    pub chunk_size: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JobResult {
    pub wall_s: f64,
    pub bytes_in: usize,
    pub bytes_out: usize,
    pub digest: String,
    pub round_trip_ok: bool,
}

/// Background job state shared across slices: which chunks are done and each output's digest and
/// length (the bytes themselves are not kept, so a large job stays small in memory). The first and
/// last chunk are decompressed and compared with their input as soon as they are compressed.
pub struct Job {
    spec: JobSpec,
    vocab: Arc<Vec<Vec<u8>>>,
    next: Arc<AtomicUsize>,
    outputs: Arc<Mutex<Vec<Option<([u8; 32], usize, bool)>>>>,
}

impl Job {
    pub fn new(spec: JobSpec) -> Job {
        Job { spec, vocab: Arc::new(vocabulary(spec.seed)), next: Arc::new(AtomicUsize::new(0)),
              outputs: Arc::new(Mutex::new(vec![None; spec.chunks])) }
    }

    pub fn done(&self) -> usize {
        self.next.load(Ordering::SeqCst).min(self.spec.chunks)
    }

    pub fn finished(&self) -> bool {
        self.done() >= self.spec.chunks
    }

    /// Work with `threads` threads until `budget` has passed (a chunk already taken is finished),
    /// or the job is done. Returns chunks completed in this slice.
    pub fn run_for(&self, threads: usize, budget: Option<Duration>) -> usize {
        let start = Instant::now();
        let before = self.done();
        std::thread::scope(|scope| {
            for _ in 0..threads.max(1) {
                scope.spawn(|| loop {
                    if budget.is_some_and(|b| start.elapsed() >= b) {
                        return;
                    }
                    let i = self.next.fetch_add(1, Ordering::SeqCst);
                    if i >= self.spec.chunks {
                        return;
                    }
                    let data = chunk(self.spec.seed, i, self.spec.chunk_size, &self.vocab);
                    let packed = compress(&data);
                    let checked = (i == 0 || i + 1 == self.spec.chunks) && decompress(&packed).as_deref() == Some(&data[..]);
                    self.outputs.lock().unwrap()[i] = Some((sha256::digest(&packed), packed.len(), checked));
                });
            }
        });
        self.done() - before
    }

    /// Digest over all chunk outputs in index order; None until every chunk is done.
    pub fn digest(&self) -> Option<(String, usize)> {
        let outputs = self.outputs.lock().unwrap();
        let mut h = Sha256::new();
        let mut total = 0;
        for o in outputs.iter() {
            let (d, len, _) = o.as_ref()?;
            h.update(&(*len as u64).to_le_bytes());
            h.update(d);
            total += len;
        }
        Some((sha256::hex(&h.finish()), total))
    }

    /// The round-trip check of chunk `index` (only the first and last are checked).
    pub fn round_trip(&self, index: usize) -> bool {
        self.outputs.lock().unwrap().get(index).and_then(|o| o.as_ref()).is_some_and(|o| o.2)
    }
}

/// The whole job with a fixed thread count (one isolated measurement).
pub fn run_job(spec: JobSpec, threads: usize) -> JobResult {
    let job = Job::new(spec);
    let start = Instant::now();
    job.run_for(threads, None);
    let wall_s = start.elapsed().as_secs_f64();
    let (digest, bytes_out) = job.digest().expect("every chunk done");
    let round_trip_ok = job.round_trip(0) && job.round_trip(spec.chunks - 1);
    JobResult { wall_s, bytes_in: spec.chunks * spec.chunk_size, bytes_out, digest, round_trip_ok }
}

/// Foreground latencies of one probe call, in milliseconds, one entry per request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProbeSamples {
    /// Scheduled arrival to completion: queueing behind a slow request and a late wake-up both count,
    /// because the machine answered that late (probe v2; OPT1 counted an overslept request from the wake-up).
    pub e2e_ms: Vec<f64>,
    /// Start of the work to completion: the request's own processing time.
    pub service_ms: Vec<f64>,
    /// PROBE1: scheduled arrival to the thread running (a late wake-up, or queueing behind the previous request).
    pub lateness_ms: Vec<f64>,
    /// PROBE1: generating the request's data, which `e2e` counts and `service` does not.
    pub prep_ms: Vec<f64>,
}

/// Foreground probe: requests arrive every `1/rate` s for `duration`; each compresses one
/// `size`-byte chunk, served one at a time.
pub fn probe(seed: u64, rate: f64, duration: Duration, size: usize) -> ProbeSamples {
    let mut out = ProbeSamples::default();
    if rate <= 0.0 {
        std::thread::sleep(duration);
        return out;
    }
    let vocab = vocabulary(seed);
    let start = Instant::now();
    let interval = Duration::from_secs_f64(1.0 / rate);
    let mut k = 0u32;
    loop {
        let arrival = interval * k;
        if arrival >= duration {
            break;
        }
        let now = start.elapsed();
        if now < arrival {
            std::thread::sleep(arrival - now);
        }
        let woke = start.elapsed().max(arrival);
        let data = chunk(seed ^ 0xF0F0, k as usize, size, &vocab);
        let begun = start.elapsed();
        std::hint::black_box(compress(&data));
        let (e2e, service) = request_latency_ms(arrival, begun, start.elapsed());
        out.e2e_ms.push(e2e);
        out.service_ms.push(service);
        out.lateness_ms.push((woke - arrival).as_secs_f64() * 1000.0);
        out.prep_ms.push((begun - woke).as_secs_f64() * 1000.0);
        k += 1;
    }
    let left = duration.saturating_sub(start.elapsed());
    std::thread::sleep(left);
    out
}

/// (end to end, service) in ms for one request: end to end counts from the scheduled arrival, so a late
/// wake-up (`begun` after `arrival` with no queue) is part of what the requester waited.
pub fn request_latency_ms(arrival: Duration, begun: Duration, done: Duration) -> (f64, f64) {
    ((done - arrival).as_secs_f64() * 1000.0, (done - begun).as_secs_f64() * 1000.0)
}

pub fn percentile(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rank = ((q * v.len() as f64).ceil() as usize).clamp(1, v.len());
    Some(v[rank - 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lz77_round_trips_and_compresses_the_corpus() {
        let vocab = vocabulary(7);
        for i in 0..4 {
            let data = chunk(7, i, 50_000, &vocab);
            let packed = compress(&data);
            assert_eq!(decompress(&packed).unwrap(), data);
            assert!(packed.len() * 2 < data.len(), "ratio {}", data.len() as f64 / packed.len() as f64);
        }
        for data in [&b""[..], b"abc", b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", b"abcdabcdabcdabcdxyz"] {
            assert_eq!(decompress(&compress(data)).unwrap(), data);
        }
        assert!(decompress(&[3, b'a']).is_none());
        assert!(decompress(&[0, 4, 9]).is_none());
    }

    #[test]
    fn the_output_digest_does_not_depend_on_threads_or_slicing() {
        let spec = JobSpec { seed: 3, chunks: 24, chunk_size: 16_384 };
        let one = run_job(spec, 1);
        let four = run_job(spec, 4);
        assert_eq!(one.digest, four.digest);
        assert!(one.round_trip_ok && four.round_trip_ok);
        let job = Job::new(spec);
        while !job.finished() {
            job.run_for(2, Some(Duration::from_millis(1)));
        }
        assert_eq!(job.digest().unwrap().0, one.digest);
        assert_ne!(run_job(JobSpec { seed: 4, ..spec }, 2).digest, one.digest);
    }

    #[test]
    fn probe_and_percentile() {
        let lat = probe(1, 50.0, Duration::from_millis(100), 4096);
        assert!(!lat.e2e_ms.is_empty() && lat.e2e_ms.len() == lat.service_ms.len());
        // end to end includes the service time and any wait since the scheduled arrival
        assert!(lat.e2e_ms.iter().zip(&lat.service_ms).all(|(e, s)| *s >= 0.0 && e >= s));
        // PROBE1: e2e is exactly lateness + prep + service, each part non-negative
        for i in 0..lat.e2e_ms.len() {
            let (l, p, s) = (lat.lateness_ms[i], lat.prep_ms[i], lat.service_ms[i]);
            assert!(l >= 0.0 && p >= 0.0 && (l + p + s - lat.e2e_ms[i]).abs() < 1e-6, "{l} + {p} + {s} != {}", lat.e2e_ms[i]);
        }
        // OPT1 counted an overslept request from its wake-up (2 ms here); probe v2 from its arrival
        let ms = Duration::from_millis;
        assert_eq!(request_latency_ms(ms(10), ms(14), ms(16)), (6.0, 2.0));
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 0.95), Some(4.0));
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 0.5), Some(2.0));
        assert_eq!(percentile(&[], 0.5), None);
    }
}
