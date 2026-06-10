#include "nmoe/tokenizer.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    kTokenTextCap = 1024,
    kPieceCap = 8192,
};

typedef struct {
    char *text;
    uint16_t length;
    uint32_t id;
} TokenRecord;

typedef struct {
    char *pair;
    uint16_t length;
} MergeRecord;

typedef struct {
    char *text;
    uint16_t length;
    uint32_t id;
} SpecialRecord;

struct nmoe_tokenizer {
    TokenRecord *records;
    uint32_t record_count;
    MergeRecord *merges;
    uint32_t merge_count;
    SpecialRecord *specials;
    uint32_t special_count;
    uint32_t byte_to_codepoint[256];
};

struct nmoe_vocab {
    char **tokens;
    int count;
};

static int read_u32_le(FILE *f, uint32_t *value) {
    return fread(value, 4, 1, f) == 1 ? 0 : -1;
}

static int read_u16_le(FILE *f, uint16_t *value) {
    return fread(value, 2, 1, f) == 1 ? 0 : -1;
}

static void init_byte_map(struct nmoe_tokenizer *tok) {
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if ((b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)) {
            tok->byte_to_codepoint[b] = (uint32_t)b;
        } else {
            tok->byte_to_codepoint[b] = 256u + (uint32_t)n++;
        }
    }
}

static int encode_codepoint(uint32_t cp, char *dst) {
    if (cp < 0x80u) {
        dst[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800u) {
        dst[0] = (char)(0xC0u | (cp >> 6));
        dst[1] = (char)(0x80u | (cp & 0x3Fu));
        return 2;
    }
    if (cp < 0x10000u) {
        dst[0] = (char)(0xE0u | (cp >> 12));
        dst[1] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
        dst[2] = (char)(0x80u | (cp & 0x3Fu));
        return 3;
    }
    dst[0] = (char)(0xF0u | (cp >> 18));
    dst[1] = (char)(0x80u | ((cp >> 12) & 0x3Fu));
    dst[2] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
    dst[3] = (char)(0x80u | (cp & 0x3Fu));
    return 4;
}

static int bytes_to_symbol_text(const struct nmoe_tokenizer *tok, const uint8_t *input, int input_len, char *out, int out_cap) {
    int written = 0;
    for (int i = 0; i < input_len && written + 4 < out_cap; ++i) {
        written += encode_codepoint(tok->byte_to_codepoint[input[i]], out + written);
    }
    out[written] = '\0';
    return written;
}

static int utf8_width(unsigned char lead) {
    if (lead < 0x80u) return 1;
    if ((lead & 0xE0u) == 0xC0u) return 2;
    if ((lead & 0xF0u) == 0xE0u) return 3;
    return 4;
}

static int classify_space(unsigned char c) {
    return isspace(c) != 0;
}

static int classify_alpha(unsigned char c) {
    return isalpha(c) != 0;
}

static int scan_chunk_spans(const char *text, int text_len, int spans[][2], int span_cap) {
    int count = 0;
    int i = 0;
    while (i < text_len && count < span_cap) {
        unsigned char c = (unsigned char)text[i];

        if (classify_space(c)) {
            int j = i;
            int saw_newline = 0;
            while (j < text_len && classify_space((unsigned char)text[j])) {
                if (text[j] == '\n' || text[j] == '\r') {
                    saw_newline = 1;
                }
                ++j;
            }
            if (saw_newline || j >= text_len) {
                spans[count][0] = i;
                spans[count][1] = j;
                ++count;
                i = j;
                continue;
            }
            if (j - i > 1) {
                spans[count][0] = i;
                spans[count][1] = j - 1;
                ++count;
                i = j - 1;
                continue;
            }
        }

        int anchor = i;
        int probe = (c == ' ' && i + 1 < text_len) ? i + 1 : i;
        if (probe < text_len) {
            unsigned char p = (unsigned char)text[probe];

            if (p == '\'' && probe + 1 < text_len) {
                char n1 = (char)(text[probe + 1] | 0x20);
                if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                    spans[count][0] = probe;
                    spans[count][1] = probe + 2;
                    ++count;
                    i = probe + 2;
                    continue;
                }
                if (probe + 2 < text_len) {
                    char n2 = (char)(text[probe + 2] | 0x20);
                    if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) {
                        spans[count][0] = probe;
                        spans[count][1] = probe + 3;
                        ++count;
                        i = probe + 3;
                        continue;
                    }
                }
            }

            if (p >= 0xC0u || classify_alpha(p)) {
                int j = probe;
                while (j < text_len) {
                    unsigned char q = (unsigned char)text[j];
                    if (q >= 0xC0u) {
                        j += utf8_width(q);
                    } else if (classify_alpha(q)) {
                        ++j;
                    } else {
                        break;
                    }
                }
                if (j > probe) {
                    spans[count][0] = anchor;
                    spans[count][1] = j;
                    ++count;
                    i = j;
                    continue;
                }
            }

            if (p >= '0' && p <= '9') {
                spans[count][0] = anchor;
                spans[count][1] = probe + 1;
                ++count;
                i = probe + 1;
                continue;
            }

            if (!classify_space(p) && !(p >= '0' && p <= '9') && !classify_alpha(p) && p < 0xC0u) {
                int j = probe;
                while (j < text_len) {
                    unsigned char q = (unsigned char)text[j];
                    if (classify_space(q) || classify_alpha(q) || (q >= '0' && q <= '9') || q >= 0xC0u) {
                        break;
                    }
                    ++j;
                }
                while (j < text_len && (text[j] == '\n' || text[j] == '\r')) {
                    ++j;
                }
                spans[count][0] = anchor;
                spans[count][1] = j;
                ++count;
                i = j;
                continue;
            }
        }

        spans[count][0] = i;
        spans[count][1] = i + 1;
        ++count;
        ++i;
    }
    return count;
}

static uint32_t find_token_id(const struct nmoe_tokenizer *tok, const char *key, uint16_t len) {
    for (uint32_t i = 0; i < tok->record_count; ++i) {
        if (tok->records[i].length == len && memcmp(tok->records[i].text, key, len) == 0) {
            return tok->records[i].id;
        }
    }
    return 0xFFFFFFFFu;
}

static uint32_t find_merge_rank(const struct nmoe_tokenizer *tok, const char *left, uint16_t left_len, const char *right, uint16_t right_len) {
    uint16_t pair_len = (uint16_t)(left_len + 1u + right_len);
    char pair[kTokenTextCap * 2 + 1];
    if (pair_len >= sizeof(pair)) {
        return 0xFFFFFFFFu;
    }
    memcpy(pair, left, left_len);
    pair[left_len] = (char)0xFF;
    memcpy(pair + left_len + 1u, right, right_len);

    for (uint32_t i = 0; i < tok->merge_count; ++i) {
        if (tok->merges[i].length == pair_len && memcmp(tok->merges[i].pair, pair, pair_len) == 0) {
            return i;
        }
    }
    return 0xFFFFFFFFu;
}

static int match_special_prefix(const struct nmoe_tokenizer *tok, const char *text, int text_len, uint32_t *out_id) {
    int best_len = 0;
    uint32_t best_id = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < tok->special_count; ++i) {
        int len = (int)tok->specials[i].length;
        if (len > best_len && len <= text_len && memcmp(text, tok->specials[i].text, (size_t)len) == 0) {
            best_len = len;
            best_id = tok->specials[i].id;
        }
    }
    if (best_len > 0) {
        *out_id = best_id;
    }
    return best_len;
}

static int find_next_special_cut(const struct nmoe_tokenizer *tok, const char *text, int start, int text_len) {
    int cut = text_len;
    for (uint32_t i = 0; i < tok->special_count; ++i) {
        int len = (int)tok->specials[i].length;
        if (len <= 0 || start + len > text_len) {
            continue;
        }
        for (int pos = start + 1; pos <= text_len - len; ++pos) {
            if (memcmp(text + pos, tok->specials[i].text, (size_t)len) == 0) {
                if (pos < cut) {
                    cut = pos;
                }
                break;
            }
        }
    }
    return cut;
}

static int utf8_piecewise_merge(const struct nmoe_tokenizer *tok, const char *text, int text_len, uint32_t *out_ids, int max_ids) {
    if (text_len <= 0) {
        return 0;
    }

    typedef struct {
        char *ptr;
        uint16_t len;
        int next;
    } Node;

    Node nodes[kPieceCap];
    int node_count = 0;
    int cursor = 0;
    while (cursor < text_len && node_count < kPieceCap) {
        unsigned char lead = (unsigned char)text[cursor];
        int step = utf8_width(lead);
        if (cursor + step > text_len) {
            step = text_len - cursor;
        }
        nodes[node_count].ptr = (char *)text + cursor;
        nodes[node_count].len = (uint16_t)step;
        nodes[node_count].next = node_count + 1;
        ++node_count;
        cursor += step;
    }
    if (node_count == 0) {
        return 0;
    }
    nodes[node_count - 1].next = -1;

    char arena[16 * 1024];
    int arena_used = 0;
    int live = node_count;

    while (live > 1) {
        uint32_t best_rank = 0xFFFFFFFFu;
        int best_node = -1;
        for (int i = 0; i != -1; i = nodes[i].next) {
            int j = nodes[i].next;
            if (j == -1) {
                break;
            }
            uint32_t rank = find_merge_rank(tok, nodes[i].ptr, nodes[i].len, nodes[j].ptr, nodes[j].len);
            if (rank < best_rank) {
                best_rank = rank;
                best_node = i;
            }
        }

        if (best_node < 0) {
            break;
        }

        int right = nodes[best_node].next;
        uint16_t merged_len = (uint16_t)(nodes[best_node].len + nodes[right].len);
        if (merged_len > kTokenTextCap) {
            break;
        }

        if (nodes[best_node].ptr + nodes[best_node].len == nodes[right].ptr) {
            nodes[best_node].len = merged_len;
        } else {
            if (arena_used + merged_len > (int)sizeof(arena)) {
                arena_used = 0;
            }
            memcpy(arena + arena_used, nodes[best_node].ptr, nodes[best_node].len);
            memcpy(arena + arena_used + nodes[best_node].len, nodes[right].ptr, nodes[right].len);
            nodes[best_node].ptr = arena + arena_used;
            nodes[best_node].len = merged_len;
            arena_used += merged_len;
        }

        nodes[best_node].next = nodes[right].next;
        --live;
    }

    int written = 0;
    for (int i = 0; i != -1 && written < max_ids; i = nodes[i].next) {
        uint32_t id = find_token_id(tok, nodes[i].ptr, nodes[i].len);
        if (id != 0xFFFFFFFFu) {
            out_ids[written++] = id;
            continue;
        }

        for (uint16_t j = 0; j < nodes[i].len && written < max_ids; ++j) {
            char one[5];
            uint32_t cp = tok->byte_to_codepoint[(uint8_t)nodes[i].ptr[j]];
            int one_len = encode_codepoint(cp, one);
            uint32_t byte_id = find_token_id(tok, one, (uint16_t)one_len);
            if (byte_id != 0xFFFFFFFFu) {
                out_ids[written++] = byte_id;
            }
        }
    }
    return written;
}

static void free_tokenizer_parts(struct nmoe_tokenizer *tok) {
    if (tok == NULL) {
        return;
    }
    for (uint32_t i = 0; i < tok->record_count; ++i) {
        free(tok->records[i].text);
    }
    free(tok->records);

    for (uint32_t i = 0; i < tok->merge_count; ++i) {
        free(tok->merges[i].pair);
    }
    free(tok->merges);

    for (uint32_t i = 0; i < tok->special_count; ++i) {
        free(tok->specials[i].text);
    }
    free(tok->specials);
}

nmoe_tokenizer *nmoe_tokenizer_load(const char *path, int quiet) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "bpe_load: cannot open %s\n", path);
        return NULL;
    }

    char magic[4];
    uint32_t version = 0;
    uint32_t vocab_size = 0;
    uint32_t merge_count = 0;
    uint32_t special_count = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "BPET", 4) != 0 ||
        read_u32_le(f, &version) || version != 1u ||
        read_u32_le(f, &vocab_size) ||
        read_u32_le(f, &merge_count) ||
        read_u32_le(f, &special_count)) {
        fclose(f);
        fprintf(stderr, "bpe_load: parse error in %s\n", path);
        return NULL;
    }

    nmoe_tokenizer *tok = calloc(1, sizeof(*tok));
    init_byte_map(tok);
    tok->record_count = vocab_size;
    tok->merge_count = merge_count;
    tok->special_count = special_count;

    tok->records = calloc(vocab_size, sizeof(TokenRecord));
    tok->merges = calloc(merge_count, sizeof(MergeRecord));
    tok->specials = calloc(special_count, sizeof(SpecialRecord));

    for (uint32_t i = 0; i < vocab_size; ++i) {
        if (read_u32_le(f, &tok->records[i].id) || read_u16_le(f, &tok->records[i].length)) goto parse_fail;
        tok->records[i].text = malloc((size_t)tok->records[i].length + 1u);
        if (fread(tok->records[i].text, 1, tok->records[i].length, f) != tok->records[i].length) goto parse_fail;
        tok->records[i].text[tok->records[i].length] = '\0';
    }

    for (uint32_t i = 0; i < merge_count; ++i) {
        uint16_t left_len = 0;
        uint16_t right_len = 0;
        if (read_u16_le(f, &left_len)) goto parse_fail;
        char *left = malloc((size_t)left_len + 1u);
        if (fread(left, 1, left_len, f) != left_len) {
            free(left);
            goto parse_fail;
        }
        left[left_len] = '\0';

        if (read_u16_le(f, &right_len)) {
            free(left);
            goto parse_fail;
        }
        char *right = malloc((size_t)right_len + 1u);
        if (fread(right, 1, right_len, f) != right_len) {
            free(left);
            free(right);
            goto parse_fail;
        }
        right[right_len] = '\0';

        tok->merges[i].length = (uint16_t)(left_len + 1u + right_len);
        tok->merges[i].pair = malloc((size_t)tok->merges[i].length + 1u);
        memcpy(tok->merges[i].pair, left, left_len);
        tok->merges[i].pair[left_len] = (char)0xFF;
        memcpy(tok->merges[i].pair + left_len + 1u, right, right_len);
        tok->merges[i].pair[tok->merges[i].length] = '\0';

        free(left);
        free(right);
    }

    for (uint32_t i = 0; i < special_count; ++i) {
        if (read_u32_le(f, &tok->specials[i].id) || read_u16_le(f, &tok->specials[i].length)) goto parse_fail;
        tok->specials[i].text = malloc((size_t)tok->specials[i].length + 1u);
        if (fread(tok->specials[i].text, 1, tok->specials[i].length, f) != tok->specials[i].length) goto parse_fail;
        tok->specials[i].text[tok->specials[i].length] = '\0';
    }

    fclose(f);
    if (!quiet) {
        fprintf(stderr, "bpe_load: %u vocab, %u merges, %u added tokens\n", vocab_size, merge_count, special_count);
    }
    return tok;

parse_fail:
    fclose(f);
    nmoe_tokenizer_free(tok);
    fprintf(stderr, "bpe_load: parse error in %s\n", path);
    return NULL;
}

void nmoe_tokenizer_free(nmoe_tokenizer *tok) {
    if (tok == NULL) {
        return;
    }
    free_tokenizer_parts(tok);
    free(tok);
}

nmoe_vocab *nmoe_vocab_load(const char *path, int quiet) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "ERROR: Cannot open vocab %s\n", path);
        return NULL;
    }

    uint32_t num_entries = 0;
    uint32_t max_id = 0;
    if (read_u32_le(f, &num_entries) || read_u32_le(f, &max_id)) {
        fclose(f);
        return NULL;
    }

    nmoe_vocab *v = calloc(1, sizeof(*v));
    v->count = (int)num_entries;
    v->tokens = calloc(num_entries, sizeof(char *));

    for (uint32_t i = 0; i < num_entries; ++i) {
        uint16_t len = 0;
        if (read_u16_le(f, &len)) {
            fclose(f);
            nmoe_vocab_free(v);
            return NULL;
        }
        if (len > 0) {
            v->tokens[i] = malloc((size_t)len + 1u);
            if (fread(v->tokens[i], 1, len, f) != len) {
                fclose(f);
                nmoe_vocab_free(v);
                return NULL;
            }
            v->tokens[i][len] = '\0';
        }
    }

    fclose(f);
    if (!quiet) {
        fprintf(stderr, "vocab: Loaded %d tokens\n", v->count);
    }
    return v;
}

void nmoe_vocab_free(nmoe_vocab *vocab) {
    if (vocab == NULL) {
        return;
    }
    for (int i = 0; i < vocab->count; ++i) {
        free(vocab->tokens[i]);
    }
    free(vocab->tokens);
    free(vocab);
}

const char *nmoe_vocab_decode_token(const nmoe_vocab *vocab, int token_id) {
    if (vocab == NULL || token_id < 0 || token_id >= vocab->count || vocab->tokens[token_id] == NULL) {
        return "<unk>";
    }
    return vocab->tokens[token_id];
}

int nmoe_tokenizer_encode(nmoe_tokenizer *tok, const char *text, uint32_t *out_ids, int max_ids) {
    if (tok == NULL || text == NULL || out_ids == NULL || max_ids <= 0) {
        return 0;
    }

    int text_len = (int)strlen(text);
    int emitted = 0;
    int cursor = 0;

    while (cursor < text_len && emitted < max_ids) {
        uint32_t special_id = 0xFFFFFFFFu;
        int special_len = match_special_prefix(tok, text + cursor, text_len - cursor, &special_id);
        if (special_len > 0) {
            out_ids[emitted++] = special_id;
            cursor += special_len;
            continue;
        }

        int segment_end = find_next_special_cut(tok, text, cursor, text_len);
        int segment_len = segment_end - cursor;
        if (segment_len <= 0) {
            ++cursor;
            continue;
        }

        int spans[kPieceCap][2];
        int span_count = scan_chunk_spans(text + cursor, segment_len, spans, kPieceCap);
        char buffer[kTokenTextCap * 4];
        for (int i = 0; i < span_count && emitted < max_ids; ++i) {
            const char *start = text + cursor + spans[i][0];
            int len = spans[i][1] - spans[i][0];
            int encoded_len = bytes_to_symbol_text(tok, (const uint8_t *)start, len, buffer, (int)sizeof(buffer));
            emitted += utf8_piecewise_merge(tok, buffer, encoded_len, out_ids + emitted, max_ids - emitted);
        }
        cursor = segment_end;
    }

    return emitted;
}
