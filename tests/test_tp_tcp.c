/* TCP gates must not depend on the requested socket-buffer size. */
#include "../ds4_tp.c"
#include <assert.h>

typedef struct {
    ds4_tp tp;
    unsigned rank;
    bool nonblocking;
    int result;
} tcp_peer;

static void pattern(void *buffer, size_t bytes, unsigned rank) {
    unsigned char *p = buffer;
    for (size_t i = 0; i < bytes; i++) p[i] = (unsigned char)(i * 37 + rank);
}

static void check_pattern(const void *buffer, size_t bytes, unsigned rank) {
    const unsigned char *p = buffer;
    for (size_t i = 0; i < bytes; i++) assert(p[i] == (unsigned char)(i * 37 + rank));
}

static void *exchange(void *arg) {
    tcp_peer *p = arg;
    ds4_tp *tp = &p->tp;
    if (p->rank) usleep(20000);
    for (unsigned step = 0; step < 160; step++) {
        const unsigned layer = (step % 80) / 2, gate = step % 2;
        pattern(tp->slab + ds4_tp_slab_out_offset(tp, layer, gate), tp->vec_bytes, p->rank);
        if (!ds4_tp_gate_exchange(tp, layer, gate, step + 1)) return NULL;
        check_pattern(tp->slab + ds4_tp_slab_in_offset(tp, layer, gate), tp->vec_bytes, 1 - p->rank);
    }
    for (unsigned rows = 1; rows <= DS4_TP_BATCH_MAX_ROWS; rows++) {
        const size_t bytes = tp->vec_bytes * rows;
        pattern(tp->slab + ds4_tp_slab_batch_out_offset(tp, 0), bytes, p->rank);
        if (!ds4_tp_batch_gate_exchange(tp, 0, rows, rows)) return NULL;
        check_pattern(tp->slab + ds4_tp_slab_batch_in_offset(tp, 0), bytes, 1 - p->rank);
    }
    const size_t capacity = 16 * 1024 * 1024 + 13;
    void *out = malloc(capacity), *in = malloc(capacity);
    assert(out && in);
    const size_t lengths[] = {1, 7, 16385, 2 * 1024 * 1024, capacity};
    p->result = 1;
    for (unsigned i = 0; i < sizeof(lengths) / sizeof(*lengths); i++) {
        pattern(out, lengths[i], p->rank);
        /* Bulk framing requires blocking I/O. An already-nonblocking socket
         * only exercises the exchange helper's descriptor-mode preservation. */
        const int ok = p->nonblocking ?
            tp_tcp_exchange(tp, NULL, NULL, out, in, lengths[i]) :
            ds4_tp_big_gate_exchange(tp, 0, i, out, in, lengths[i]);
        if (!ok) {
            p->result = 0;
            break;
        }
        check_pattern(in, lengths[i], 1 - p->rank);
    }
    free(in); free(out);
    return NULL;
}

static void pair(int fd[2], bool tcp) {
    if (tcp) {
        int listener = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in addr = {.sin_family = AF_INET,
            .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
        socklen_t len = sizeof(addr);
        assert(listener >= 0 && bind(listener, (struct sockaddr *)&addr, len) == 0);
#ifdef __linux__
        /* Loopback's 32 KiB segments exceed the tiny receive window below.
         * Use small segments to test backpressure without zero-window probes
         * throttling a 16 MiB transfer for minutes. Set this before SYN. */
        const int mss = 512;
        assert(setsockopt(listener, IPPROTO_TCP, TCP_MAXSEG, &mss, sizeof(mss)) == 0);
#endif
        assert(getsockname(listener, (struct sockaddr *)&addr, &len) == 0);
        assert(listen(listener, 1) == 0);
        fd[0] = socket(AF_INET, SOCK_STREAM, 0);
        assert(fd[0] >= 0);
#ifdef __linux__
        assert(setsockopt(fd[0], IPPROTO_TCP, TCP_MAXSEG, &mss, sizeof(mss)) == 0);
#endif
        assert(connect(fd[0], (struct sockaddr *)&addr, len) == 0);
        fd[1] = accept(listener, NULL, NULL);
        assert(fd[1] >= 0);
        close(listener);
    } else {
        assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    }
    for (unsigned i = 0; i < 2; i++) {
        int small = 1024;
        assert(setsockopt(fd[i], SOL_SOCKET, SO_SNDBUF, &small, sizeof(small)) == 0);
        assert(setsockopt(fd[i], SOL_SOCKET, SO_RCVBUF, &small, sizeof(small)) == 0);
#ifdef SO_NOSIGPIPE
        int one = 1;
        assert(setsockopt(fd[i], SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one)) == 0);
#endif
        assert(tp_socket_set_gate_timeout(fd[i], 1000));
        if (tcp) {
            int one = 1;
            assert(setsockopt(fd[i], IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) == 0);
        }
    }
}

static void check_stall_and_close(bool tcp) {
    int fd[2];
    pair(fd, tcp);
    ds4_tp tp = {.data_fd = fd[0], .gate_timeout_ms = 50};
    const int flags = fcntl(fd[0], F_GETFL);
    assert(flags >= 0 && !(flags & O_NONBLOCK));
    char out[32768] = {0}, in[32768];
    double start = tp_now_sec();
    assert(!tp_tcp_exchange(&tp, NULL, NULL, out, in, sizeof(out)));
    assert(errno == ETIMEDOUT && tp_now_sec() - start < 1.0);
    assert(fcntl(fd[0], F_GETFL) == flags);
    close(fd[0]); close(fd[1]);
    pair(fd, tcp);
    tp.data_fd = fd[0];
    assert(shutdown(fd[1], SHUT_WR) == 0);
    assert(!tp_tcp_exchange(&tp, NULL, NULL, out, in, sizeof(out)));
    assert(errno == ECONNRESET);
    assert(fcntl(fd[0], F_GETFL) == flags);
    close(fd[1]);
    assert(!tp_tcp_exchange(&tp, NULL, NULL, out, in, sizeof(out)));
    close(fd[0]);
    puts("TCP stalled, half-closed and disconnected peers: PASS");
}

static void check_exchange(bool tcp, bool nonblocking) {
    int fd[2];
    pair(fd, tcp);
    int flags[2];
    tcp_peer peer[2] = {0};
    pthread_t threads[2];
    for (unsigned i = 0; i < 2; i++) {
        flags[i] = fcntl(fd[i], F_GETFL);
        assert(flags[i] >= 0);
        if (nonblocking) {
            flags[i] |= O_NONBLOCK;
            assert(fcntl(fd[i], F_SETFL, flags[i]) == 0);
        }
        peer[i].rank = i;
        peer[i].nonblocking = nonblocking;
        peer[i].tp = (ds4_tp){.data_fd = fd[i], .n_layer = 40, .n_slots = 80,
            .n_embd = 5120, .vec_bytes = 5120 * 4, .gate_timeout_ms = 1000};
        tp_slab_layout(&peer[i].tp);
        peer[i].tp.slab = calloc(1, peer[i].tp.slab_bytes);
        assert(peer[i].tp.slab);
        assert(pthread_create(&threads[i], NULL, exchange, &peer[i]) == 0);
    }
    for (unsigned i = 0; i < 2; i++) {
        assert(pthread_join(threads[i], NULL) == 0 && peer[i].result);
        assert(fcntl(fd[i], F_GETFL) == flags[i]);
        free(peer[i].tp.slab);
        close(fd[i]);
    }
    printf("%s tiny-buffer decode, 1..8-row gates and bulk payloads, nonblocking=%d: PASS\n",
           tcp ? "TCP" : "Unix", nonblocking);
}


typedef struct { ds4_tp tp; FILE *fp; uint64_t bytes; int ok; } payload_peer;
static void *send_snapshot(void *arg) {
    payload_peer *p = arg;
    p->ok = tp_payload_transfer(&p->tp, p->fp, p->bytes, true);
    const uint32_t sentinel = 0x12345678;
    if (p->ok) p->ok = tp_write_full(p->tp.control_fd, &sentinel, sizeof(sentinel));
    return NULL;
}
static void check_payload_staging(bool disk_full) {
    int fd[2]; pair(fd, true);
    const size_t bytes = 3u * 1024u * 1024u + 37u;
    unsigned char *data = malloc(bytes); assert(data); pattern(data, bytes, 7);
    FILE *src = tmpfile(), *dst = disk_full ? fopen("/dev/full", "wb") : tmpfile();
    assert(src && dst && fwrite(data, 1, bytes, src) == bytes); rewind(src);
    payload_peer tx = {.tp = {.control_fd = fd[0]}, .fp = src, .bytes = bytes};
    ds4_tp rx = {.control_fd = fd[1]}; pthread_t thread;
    assert(!pthread_create(&thread, NULL, send_snapshot, &tx));
    assert(tp_payload_transfer(&rx, dst, bytes, false) == !disk_full);
    uint32_t sentinel = 0;
    assert(tp_read_full(fd[1], &sentinel, sizeof(sentinel)) && sentinel == 0x12345678);
    assert(!pthread_join(thread, NULL) && tx.ok);
    if (!disk_full) { rewind(dst); assert(fread(data, 1, bytes, dst) == bytes); check_pattern(data, bytes, 7); }
    fclose(src); fclose(dst); free(data); close(fd[0]); close(fd[1]);
    puts(disk_full ? "Snapshot disk-full drain preserves framing: PASS" : "Multi-chunk snapshot bytes: PASS");
}

#ifdef __linux__
#include <sys/wait.h>

/* Exercise actual cross-process mapping, slot reuse, negotiation fallback,
 * one-sided import failure, and a peer disappearing inside a gate. */
static void check_local_processes(int mode) {
    int data[2], control[2];
    pair(data, false); pair(control, false);
    const pid_t pid = fork();
    assert(pid >= 0);
    const int rank = pid == 0;
    close(data[1-rank]); close(control[1-rank]);
    ds4_tp tp = {.rank = rank, .data_fd = data[rank], .control_fd = control[rank],
        .n_layer = 2, .n_slots = 4, .local_capable = true, .gate_timeout_ms = 100};
    unsigned char handle[64], peer_handle[64];
    pattern(handle, sizeof(handle), rank);
    const int expected = mode == 1 || mode == 4 ? 0 : 1;
    if (mode == 4) setenv("DS4_CUDA_TP_IPC", "0", 1);
    assert(ds4_tp_local_setup(&tp, !(mode == 1 && rank == 1), handle, peer_handle) == expected);
    if (expected) {
        check_pattern(peer_handle, sizeof(peer_handle), 1-rank);
        const int confirmed = ds4_tp_local_confirm(&tp, !(mode == 2 && rank == 1));
        assert(confirmed == (mode != 2));
        if (confirmed) {
            unsigned char events[128], peer_events[128];
            for (unsigned i = 0; i < sizeof(events); i++) events[i] = (unsigned char)(rank * 97u + i);
            for (unsigned trial = 0; trial < 3; trial++) {
                const int ready = !(trial == 1 && rank == 1);
                const int setup = ds4_tp_local_event_setup(&tp, ready, events, peer_events);
                assert(setup == (trial != 1));
                if (setup) {
                    for (unsigned i = 0; i < sizeof(events); i++)
                        assert(peer_events[i] == (unsigned char)((1-rank) * 97u + i));
                    assert(ds4_tp_local_event_confirm(&tp, !(trial == 2 && rank == 1)) == (trial != 2));
                }
                assert(ds4_tp_is_local(&tp));
            }
            for (uint64_t seq = 1; seq <= 2048; seq++) {
                if (rank == 1 && seq % 137 == 0) usleep(100);
                assert(ds4_tp_local_gate(&tp, ((seq-1)%4)/2, (seq-1)%2, seq));
            }
            /* Both processes must leave the last successful gate before a
             * normal teardown (or the intentional peer-death test) starts. */
            uint8_t finished = 1, peer_finished = 0;
            assert(tp_write_full(control[rank], &finished, sizeof(finished)));
            assert(tp_read_full(control[rank], &peer_finished, sizeof(peer_finished)) && peer_finished == 1);
            if (mode == 3 && rank == 1) {
                /* A vanished process cannot publish its normal close flag. */
                close(control[rank]); close(data[rank]); _exit(0);
            }
            if (mode == 3) assert(!ds4_tp_local_gate(&tp, 0, 0, 2049));
        }
        tp_local_close(&tp);
    }
    if (mode == 4) unsetenv("DS4_CUDA_TP_IPC");
    close(data[rank]); close(control[rank]);
    if (rank == 1) _exit(0);
    int status;
    assert(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && !WEXITSTATUS(status));
    printf("Local transport cross-process mode=%d: PASS\n", mode);
}
#endif

typedef struct {
    ds4_tp tp;
    int *tokens;
    ds4_vision_span *images;
    uint32_t n_tokens, n_images;
    int ok;
} image_peer;

static void *send_images(void *ud) {
    image_peer *p = ud;
    p->ok = ds4_tp_send_sync_multimodal(&p->tp, 77, p->tokens, p->n_tokens,
                                       p->images, p->n_images) &&
            ds4_tp_send_stop(&p->tp);
    return NULL;
}

static void check_image_stream(bool large) {
    const uint32_t n_images = large ? 1023 : 6144;
    const uint32_t rows = large ? 1024 : 2;
    const uint32_t dim = large ? 5120 : 8;
    const uint32_t n_tokens = n_images * (rows + (large ? 0 : 1)) + 16;
    const size_t image_bytes = (size_t)rows * dim * sizeof(float);
    int fd[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
    image_peer tx = {.tp = {.control_fd = fd[0], .peer_ctx = 0, .n_embd = dim},
        .n_tokens = n_tokens, .n_images = n_images};
    ds4_tp rx = {.control_fd = fd[1], .peer_ctx = 1048576, .n_embd = dim};
    tx.tokens = calloc(n_tokens, sizeof(*tx.tokens));
    tx.images = calloc(n_images, sizeof(*tx.images));
    /* The large test reuses an immutable source block: no huge sender copy is
     * needed. The receiver still allocates and verifies the full ~20 GiB. */
    uint8_t *data = malloc(large ? image_bytes : image_bytes * n_images);
    assert(tx.tokens && tx.images && data);
    pattern(data, large ? image_bytes : image_bytes * n_images, 7);
    for (uint32_t i = 0; i < n_tokens; i++) tx.tokens[i] = (int)(i % 129280);
    for (uint32_t i = 0; i < n_images; i++) {
        tx.images[i].token_start = i * (rows + (large ? 0 : 1));
        tx.images[i].embedding = (ds4_vision_embedding){
            .data = (float *)(data + (large ? 0 : i * image_bytes)),
            .token_count = rows, .width = 1000, .height = 800,
            .content_width = 980, .content_height = 780};
        memcpy(tx.images[i].embedding.fingerprint, &i, sizeof(i));
    }
    /* Reject invalid spans before emitting any bytes, so the pair remains usable. */
    tx.images[1].token_start = 0;
    assert(!ds4_tp_send_sync_multimodal(&tx.tp, 77, tx.tokens, n_tokens, tx.images, n_images));
    tx.images[1].token_start = rows + (large ? 0 : 1);
    pthread_t thread;
    const double start = tp_now_sec();
    assert(!pthread_create(&thread, NULL, send_images, &tx));
    char err[256] = {0};
    ds4_tp_command cmd;
    assert(ds4_tp_recv_command(&rx, &cmd, err, sizeof(err)));
    assert(cmd.type == DS4_TP_FRAME_SYNC_MULTIMODAL && cmd.session_id == 77);
    assert(cmd.n_tokens == n_tokens && cmd.n_images == n_images);
    assert(!memcmp(cmd.tokens, tx.tokens, (size_t)n_tokens * sizeof(*tx.tokens)));
    for (uint32_t i = 0; i < n_images; i++) {
        const ds4_vision_embedding *a = &tx.images[i].embedding, *b = &cmd.images[i].embedding;
        assert(cmd.images[i].token_start == tx.images[i].token_start);
        assert(a->token_count == b->token_count && a->width == b->width &&
               a->height == b->height && a->content_width == b->content_width &&
               a->content_height == b->content_height);
        assert(!memcmp(a->fingerprint, b->fingerprint, sizeof(a->fingerprint)));
        assert(!memcmp(a->data, b->data, image_bytes));
    }
    ds4_tp_command_free(&cmd);
    assert(ds4_tp_recv_command(&rx, &cmd, err, sizeof(err)) && cmd.type == DS4_TP_FRAME_STOP);
    ds4_tp_command_free(&cmd);
    assert(!pthread_join(thread, NULL) && tx.ok);
    free(tx.images); free(tx.tokens); free(data); close(fd[0]); close(fd[1]);
    printf("Multimodal stream: %u images, %u tokens, %.3f GiB exact features + next command: PASS (%.3fs)\n",
           n_images, n_tokens, (double)image_bytes * n_images / (1ull << 30), tp_now_sec() - start);
}

static void check_image_stream_rejection(void) {
    for (int mode = 0; mode < 6; mode++) {
        int fd[2]; assert(socketpair(AF_UNIX, SOCK_STREAM, 0, fd) == 0);
        ds4_tp rx = {.control_fd = fd[0], .peer_ctx = 32, .n_embd = 8};
        ds4_tp_multimodal_command_header h = {77, mode == 0 ? 33 : 4, mode == 1 ? 5 : 1};
        assert(tp_send_frame(fd[1], DS4_TP_FRAME_SYNC_MULTIMODAL, &h, sizeof(h)));
        if (mode >= 2) {
            int32_t tokens[4] = {1, 2, 3, 4};
            assert(tp_write_full(fd[1], tokens, sizeof(tokens)));
            ds4_tp_vision_wire wire = {.token_start = 1, .token_count = 2,
                .data_width = 8, .width = 100, .height = 100};
            if (mode == 2) wire.data_width = 9;
            if (mode == 3) wire.token_count = 4;
            if (mode == 4) wire.width = 0;
            assert(tp_write_full(fd[1], &wire, sizeof(wire)));
            /* mode 5 leaves the otherwise valid image feature stream truncated. */
        }
        shutdown(fd[1], SHUT_WR);
        ds4_tp_command cmd; char err[128] = {0};
        assert(!ds4_tp_recv_command(&rx, &cmd, err, sizeof(err)) && ds4_tp_failed(&rx));
        close(fd[0]); close(fd[1]);
    }
    puts("Multimodal bad count/context/dimension/span/size and truncated stream: PASS");
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "--large-images")) {
        check_image_stream(true);
        return 0;
    }
    check_image_stream(false);
    check_image_stream_rejection();
#ifdef __linux__
    for (int mode = 0; mode < 5; mode++) check_local_processes(mode);
#endif
    check_payload_staging(false);
    check_payload_staging(true);
    for (int tcp = 0; tcp < 2; tcp++) {
        check_exchange(tcp, false);
        check_exchange(tcp, true);
        check_stall_and_close(tcp);
    }
    return 0;
}
