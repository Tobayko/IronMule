    // Count tiles in the mask consumed by the sampled MMA kernel (eight queries by 256 keys).
    // Run in an instrumented process only. Its elapsed time is excluded from performance results.
    const char * path = getenv("IRONMULE_LAB_MASK_TILES");
    if (path && n_stream == 1 && n_tps >= 440 && n_tps <= 485 && n_kv % 256 == 0) {
        uint64_t total = 0, trailing = 0, below_last = 0, visible = 0;
        for (int64_t q = 0; q < n_tps; q += 8) {
            int last = -1, count = 0;
            for (int64_t k = 0; k < n_kv; k += 256) {
                bool used = false;
                for (int64_t i = q; i < std::min(q + 8, n_tps) && !used; ++i) {
                    const float * tile = data + i*n_kv + k;
                    used = std::any_of(tile, tile + 256, [](float x) { return x != -INFINITY; });
                }
                if (used) {
                    last = int(k / 256);
                    ++count;
                }
            }
            total += n_kv / 256;
            visible += count;
            trailing += n_kv / 256 - (last + 1);
            below_last += last + 1 - count;
        }
        if (FILE * out = fopen(path, "a")) {
            fprintf(out, "%lld,%lld,%llu,%llu,%llu,%llu\n",
                    (long long) n_tps, (long long) n_kv, (unsigned long long) total,
                    (unsigned long long) visible, (unsigned long long) trailing,
                    (unsigned long long) below_last);
            fclose(out);
        } else {
            GGML_ABORT("cannot save diagnostic mask tiles");
        }
    }
