// Benchmark expert pread strategies against a real layer file.
// Build: clang -O2 scripts/pread_bench.c -o /tmp/pread_bench
// Run from repo root: /tmp/pread_bench qwen36_35b/packed_experts/layer_00.bin

#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define K 8
#define EXPERTS 256

static double now_ms(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "qwen36_35b/packed_experts/layer_00.bin";
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    size_t esz = (size_t)st.st_size / EXPERTS;
    printf("expert size: %.2f MB, total read per layer: %.1f MB\n", esz / 1048576.0, K * esz / 1048576.0);

    uint8_t **bufs = malloc(K * sizeof(uint8_t *));
    for (int i = 0; i < K; i++) posix_memalign((void **)&bufs[i], 16384, esz);
    int *fds = malloc(K * sizeof(int));
    for (int i = 0; i < K; i++) fds[i] = open(path, O_RDONLY);

    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    unsigned seed = 12345;

    // Warm a working set: read a bunch of experts once so page cache is hot for them
    for (int e = 0; e < 64; e++) {
        ssize_t r = pread(fd, bufs[0], esz, (off_t)(e * esz));
        (void)r;
    }

    for (int round = 0; round < 3; round++) {
        size_t *idx = malloc(K * sizeof(size_t));
        for (int i = 0; i < K; i++) idx[i] = rand_r(&seed) % 64; // hot region

        // A: serial pread, one fd
        double t0 = now_ms();
        for (int i = 0; i < K; i++) {
            ssize_t r = pread(fd, bufs[i], esz, (off_t)(idx[i] * esz));
            (void)r;
        }
        double tA = now_ms() - t0;

        // B: 8-wide dispatch_apply, shared fd (current runtime strategy)
        t0 = now_ms();
        dispatch_apply(K, q, ^(size_t i) {
            ssize_t r = pread(fd, bufs[i], esz, (off_t)(idx[i] * esz));
            (void)r;
        });
        double tB = now_ms() - t0;

        // C: 8-wide, one fd per worker
        t0 = now_ms();
        dispatch_apply(K, q, ^(size_t i) {
            ssize_t r = pread(fds[i], bufs[i], esz, (off_t)(idx[i] * esz));
            (void)r;
        });
        double tC = now_ms() - t0;

        // D: 32-wide chunked (4 chunks per expert), shared fd
        t0 = now_ms();
        dispatch_apply(K * 4, q, ^(size_t t) {
            size_t i = t / 4, c = t % 4, chunk = esz / 4;
            ssize_t r = pread(fd, bufs[i] + c * chunk, chunk, (off_t)(idx[i] * esz + c * chunk));
            (void)r;
        });
        double tD = now_ms() - t0;

        // E: 32-wide chunked, per-worker fds
        t0 = now_ms();
        dispatch_apply(K * 4, q, ^(size_t t) {
            size_t i = t / 4, c = t % 4, chunk = esz / 4;
            ssize_t r = pread(fds[i], bufs[i] + c * chunk, chunk, (off_t)(idx[i] * esz + c * chunk));
            (void)r;
        });
        double tE = now_ms() - t0;

        double mb = K * esz / 1048576.0;
        printf("round %d (hot): A serial %6.2fms (%5.1f GB/s) | B 8-wide-1fd %6.2fms (%5.1f GB/s) | C 8-wide-8fd %6.2fms (%5.1f GB/s) | D 32chunk-1fd %6.2fms (%5.1f GB/s) | E 32chunk-8fd %6.2fms (%5.1f GB/s)\n",
               round,
               tA, mb / tA, tB, mb / tB, tC, mb / tC, tD, mb / tD, tE, mb / tE);
    }

    // G: dispatch_group_async pattern (what the runtime actually does)
    for (int round = 0; round < 3; round++) {
        size_t *idx = malloc(K * sizeof(size_t));
        for (int i = 0; i < K; i++) idx[i] = rand_r(&seed) % 64;
        dispatch_group_t group = dispatch_group_create();
        double t0 = now_ms();
        for (size_t i = 0; i < K; i++) {
            size_t ii = i;
            dispatch_group_async(group, q, ^{
                ssize_t r = pread(fd, bufs[ii], esz, (off_t)(idx[ii] * esz));
                (void)r;
            });
        }
        dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
        double tG = now_ms() - t0;
        double mb = K * esz / 1048576.0;
        printf("round %d (hot): G group_async-1fd %6.2fms (%5.1f GB/s)\n", round, tG, mb / tG);
    }

    // F: cold-ish reads from the far end of the file (likely uncached)
    {
        size_t *idx = malloc(K * sizeof(size_t));
        for (int i = 0; i < K; i++) idx[i] = 192 + (size_t)(rand_r(&seed) % 64);
        double t0 = now_ms();
        dispatch_apply(K, q, ^(size_t i) {
            ssize_t r = pread(fd, bufs[i], esz, (off_t)(idx[i] * esz));
            (void)r;
        });
        double tF = now_ms() - t0;
        double mb = K * esz / 1048576.0;
        printf("cold-ish: B 8-wide-1fd %6.2fms (%5.1f GB/s)\n", tF, mb / tF);
    }
    return 0;
}
