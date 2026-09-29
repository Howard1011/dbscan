#include "my_window.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Test events
static const int TEST_EVENTS[] = {500, 501, 502};
static const int NUM_EVENTS    = 3;

// Read emb_eventXXX.dat: 600 rows x 12 cols of floating values -> quantise to emb_t
static int read_emb(const char *fname, emb_t emb[N][EMB_DIM]) {
    FILE *fp = fopen(fname, "r");
    if (!fp) { fprintf(stderr, "ERROR: cannot open %s\n", fname); return -1; }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < EMB_DIM; j++) {
            double v;
            if (fscanf(fp, "%lf", &v) != 1) {
                fprintf(stderr, "ERROR: short read in %s at (%d,%d)\n", fname, i, j);
                fclose(fp); return -1;
            }
            emb[i][j] = (emb_t)v;
        }
    }
    fclose(fp);
    return 0;
}

// Read golden_eventXXX.dat: 600 integer labels, one per line
static int read_golden(const char *fname, int golden[N]) {
    FILE *fp = fopen(fname, "r");
    if (!fp) { fprintf(stderr, "ERROR: cannot open %s\n", fname); return -1; }
    for (int i = 0; i < N; i++) {
        if (fscanf(fp, "%d", &golden[i]) != 1) {
            fprintf(stderr, "ERROR: short read in %s at %d\n", fname, i);
            fclose(fp); return -1;
        }
    }
    fclose(fp);
    return 0;
}

// same_partition: compare two labelings ignoring the id numbering
static bool same_partition(const int a[N], const int b[N]) {
    int a2b[N], b2a[N];
    memset(a2b, -1, sizeof(a2b));
    memset(b2a, -1, sizeof(b2a));
    for (int i = 0; i < N; i++) {
        int ai = a[i], bi = b[i];
        if (a2b[ai] == -1) a2b[ai] = bi;
        if (b2a[bi] == -1) b2a[bi] = ai;
        if (a2b[ai] != bi || b2a[bi] != ai) return false;
    }
    return true;
}

int main() {
    static emb_t emb[N][EMB_DIM];
    static int   out_labels[N];
    static int   golden[N];

    int total_errors = 0;

    for (int e = 0; e < NUM_EVENTS; e++) {
        int ev = TEST_EVENTS[e];

        char emb_fname[64], golden_fname[64];
        sprintf(emb_fname,    "data/emb_event%d.dat",    ev);
        sprintf(golden_fname, "data/golden_event%d.dat", ev);

        if (read_emb(emb_fname, emb) != 0) return 1;
        if (read_golden(golden_fname, golden) != 0) return 1;

        // Call DUT
        my_window(emb, out_labels);

        // Dump actual labels produced by the DUT (one label per line)
        char out_fname[64];
        sprintf(out_fname, "data/labels_event%d.dat", ev);
        FILE *ofp = fopen(out_fname, "w");
        if (!ofp) { fprintf(stderr, "ERROR: cannot open %s for write\n", out_fname); return 1; }
        for (int i = 0; i < N; i++) fprintf(ofp, "%d\n", out_labels[i]);
        fclose(ofp);
        printf("  wrote %s\n", out_fname);

        int mismatches = 0;
        for (int i = 0; i < N; i++) {
            if (out_labels[i] != golden[i]) mismatches++;
        }
        bool partition_ok = same_partition(out_labels, golden);

        int n_clusters = 0;
        for (int i = 0; i < N; i++) if (out_labels[i] > n_clusters) n_clusters = out_labels[i];
        n_clusters++;

        printf("event %d: clusters=%d  exact_match=%s  partition_match=%s  mismatches=%d\n",
               ev, n_clusters,
               (mismatches == 0) ? "PASS" : "FAIL",
               partition_ok ? "PASS" : "FAIL",
               mismatches);

        // Contract is partition_match: same clustering, label values may differ.
        if (!partition_ok) total_errors++;
        (void)mismatches;
    }

    if (total_errors == 0) {
        printf("ALL EVENTS PASS (partition_match)\n");
        return 0;
    } else {
        printf("FAIL: %d event(s) with wrong partition\n", total_errors);
        return 1;
    }
}
