/*
 * book_builder.c — Fast orderbook reconstruction from Tardis incremental_book_L2
 *
 * Reads CSV from file (mmap) or stdin, outputs binary snapshots to stdout.
 * ~50-100x faster than the Python SortedDict approach.
 *
 * Build:  cc -O2 -o tools/book_builder tools/book_builder.c
 *
 * Binary output format (per snapshot):
 *   int64_t  timestamp_ms
 *   double   bid_price[20]
 *   double   bid_qty[20]
 *   double   ask_price[20]
 *   double   ask_qty[20]
 * Total: 8 + 80*8 = 648 bytes per snapshot
 *
 * Usage:  ./book_builder input.csv > snapshots.bin
 *    or:  cat input.csv | ./book_builder > snapshots.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#define MAX_LEVELS      200
#define SNAP_LEVELS      20
#define SNAP_INTERVAL_MS 100
#define OUT_BUF_SIZE     (648 * 4096)  /* Buffer ~4096 snapshots before flushing */

typedef struct {
    double price;
    double amount;
} Level;

typedef struct {
    Level levels[MAX_LEVELS];
    int   count;
} Book;

/* Update a price level in the book. O(n) scan, n <= MAX_LEVELS. */
static inline void book_update(Book *b, double price, double amount)
{
    Level *lev = b->levels;
    int cnt = b->count;

    for (int i = 0; i < cnt; i++) {
        if (lev[i].price == price) {
            if (amount == 0.0) {
                lev[i] = lev[--b->count];
            } else {
                lev[i].amount = amount;
            }
            return;
        }
    }
    if (amount > 0.0 && cnt < MAX_LEVELS) {
        lev[cnt].price = price;
        lev[cnt].amount = amount;
        b->count = cnt + 1;
    }
}

static int cmp_desc(const void *a, const void *b)
{
    double d = ((const Level *)b)->price - ((const Level *)a)->price;
    return (d > 0) - (d < 0);
}

static int cmp_asc(const void *a, const void *b)
{
    double d = ((const Level *)a)->price - ((const Level *)b)->price;
    return (d > 0) - (d < 0);
}

/* Fast integer parsing (for timestamps) */
static inline int64_t parse_int64(const char *p, const char **end)
{
    int64_t v = 0;
    while (*p >= '0' && *p <= '9') {
        v = v * 10 + (*p - '0');
        p++;
    }
    *end = p;
    return v;
}

/* Fast float parsing (sufficient for prices/amounts) */
static inline double parse_double(const char *p, const char **end)
{
    double v = 0.0;
    int neg = 0;

    if (*p == '-') { neg = 1; p++; }

    while (*p >= '0' && *p <= '9') {
        v = v * 10.0 + (*p - '0');
        p++;
    }

    if (*p == '.') {
        p++;
        double frac = 0.1;
        while (*p >= '0' && *p <= '9') {
            v += (*p - '0') * frac;
            frac *= 0.1;
            p++;
        }
    }

    /* Handle scientific notation (e.g., 1.5e-7) */
    if (*p == 'e' || *p == 'E') {
        p++;
        int exp_neg = 0;
        if (*p == '-') { exp_neg = 1; p++; }
        else if (*p == '+') { p++; }
        int exp = 0;
        while (*p >= '0' && *p <= '9') {
            exp = exp * 10 + (*p - '0');
            p++;
        }
        double mult = 1.0;
        for (int i = 0; i < exp; i++) mult *= 10.0;
        if (exp_neg) v /= mult; else v *= mult;
    }

    *end = p;
    return neg ? -v : v;
}

/* Skip to next comma or newline, return pointer after delimiter */
static inline const char *skip_field(const char *p)
{
    while (*p && *p != ',' && *p != '\n' && *p != '\r') p++;
    if (*p == ',') return p + 1;
    return p;
}

/* Skip to next newline, return pointer after it */
static inline const char *skip_line(const char *p, const char *end)
{
    while (p < end && *p != '\n') p++;
    if (p < end) p++;  /* skip \n */
    return p;
}

/* Read entire stdin into a malloc'd buffer */
static char *read_stdin(size_t *out_len)
{
    size_t cap = 256 * 1024 * 1024;  /* Start with 256MB */
    size_t len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;

    while (1) {
        if (len + (64 * 1024 * 1024) > cap) {
            cap *= 2;
            char *nb = realloc(buf, cap);
            if (!nb) { free(buf); return NULL; }
            buf = nb;
        }
        size_t n = fread(buf + len, 1, 64 * 1024 * 1024, stdin);
        len += n;
        if (n == 0) break;
    }
    *out_len = len;
    return buf;
}

int main(int argc, char *argv[])
{
    const char *data = NULL;
    size_t data_len = 0;
    int use_mmap = 0;
    int fd = -1;
    char *stdin_buf = NULL;

    if (argc > 1) {
        /* Memory-map the input file for maximum speed */
        fd = open(argv[1], O_RDONLY);
        if (fd < 0) { perror("open"); return 1; }

        struct stat st;
        if (fstat(fd, &st) < 0) { perror("fstat"); return 1; }
        data_len = st.st_size;

        data = mmap(NULL, data_len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) { perror("mmap"); return 1; }

        /* Advise sequential access for OS read-ahead */
        madvise((void *)data, data_len, MADV_SEQUENTIAL);
        use_mmap = 1;
    } else {
        /* Read from stdin into buffer */
        stdin_buf = read_stdin(&data_len);
        if (!stdin_buf) {
            fprintf(stderr, "Failed to read stdin\n");
            return 1;
        }
        data = stdin_buf;
    }

    /* Set up buffered output */
    setvbuf(stdout, NULL, _IOFBF, 1 << 20);

    Book bids = { .count = 0 };
    Book asks = { .count = 0 };
    int64_t last_snap_ts = 0;
    int saw_update = 0;
    unsigned char snap_buf[648];

    const char *p = data;
    const char *end = data + data_len;

    /* Skip CSV header line */
    p = skip_line(p, end);

    long lines_processed = 0;

    while (p < end) {
        const char *line_start = p;

        /* Skip exchange field */
        p = skip_field(p);
        /* Skip symbol field */
        p = skip_field(p);

        /* Parse timestamp */
        const char *after;
        int64_t ts = parse_int64(p, &after);
        if (ts > 1000000000000000LL) ts /= 1000;  /* us -> ms */
        p = after;
        if (*p == ',') p++;

        /* Skip local_timestamp */
        p = skip_field(p);

        /* Parse is_snapshot (starts with 't' for true) */
        int is_snapshot = (*p == 't');
        p = skip_field(p);

        /* Handle reconnect: snapshot after updates means reset */
        if (is_snapshot && saw_update) {
            bids.count = 0;
            asks.count = 0;
            saw_update = 0;
        } else if (!is_snapshot) {
            saw_update = 1;
        }

        /* Parse side */
        int is_bid = (*p == 'b');
        p = skip_field(p);

        /* Parse price */
        double price = parse_double(p, &after);
        p = after;
        if (*p == ',') p++;

        /* Parse amount */
        double amount = parse_double(p, &after);
        p = after;

        /* Skip to next line */
        while (p < end && *p != '\n') p++;
        if (p < end) p++;

        /* Apply update */
        book_update(is_bid ? &bids : &asks, price, amount);
        lines_processed++;

        /* Take snapshot at interval */
        if (ts - last_snap_ts >= SNAP_INTERVAL_MS
            && bids.count >= SNAP_LEVELS
            && asks.count >= SNAP_LEVELS)
        {
            /* Sort books for snapshot */
            qsort(bids.levels, bids.count, sizeof(Level), cmp_desc);
            qsort(asks.levels, asks.count, sizeof(Level), cmp_asc);

            /* Build binary snapshot */
            int off = 0;
            memcpy(snap_buf + off, &ts, 8); off += 8;

            for (int i = 0; i < SNAP_LEVELS; i++) {
                double v = (i < bids.count) ? bids.levels[i].price : 0.0;
                memcpy(snap_buf + off, &v, 8); off += 8;
            }
            for (int i = 0; i < SNAP_LEVELS; i++) {
                double v = (i < bids.count) ? bids.levels[i].amount : 0.0;
                memcpy(snap_buf + off, &v, 8); off += 8;
            }
            for (int i = 0; i < SNAP_LEVELS; i++) {
                double v = (i < asks.count) ? asks.levels[i].price : 0.0;
                memcpy(snap_buf + off, &v, 8); off += 8;
            }
            for (int i = 0; i < SNAP_LEVELS; i++) {
                double v = (i < asks.count) ? asks.levels[i].amount : 0.0;
                memcpy(snap_buf + off, &v, 8); off += 8;
            }

            fwrite(snap_buf, 1, 648, stdout);
            last_snap_ts = ts;

            /* Trim books to keep bounded */
            if (bids.count > 60) bids.count = 60;
            if (asks.count > 60) asks.count = 60;
        }
    }

    fflush(stdout);

    /* Print stats to stderr */
    fprintf(stderr, "Processed %ld lines\n", lines_processed);

    /* Cleanup */
    if (use_mmap) {
        munmap((void *)data, data_len);
        close(fd);
    } else {
        free(stdin_buf);
    }

    return 0;
}
