#ifndef NMOE_TOKENIZER_H
#define NMOE_TOKENIZER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nmoe_prompt_tokens {
    uint32_t *ids;
    int count;
} nmoe_prompt_tokens;

typedef struct nmoe_tokenizer nmoe_tokenizer;
typedef struct nmoe_vocab nmoe_vocab;

nmoe_tokenizer *nmoe_tokenizer_load(const char *path, int quiet);
void nmoe_tokenizer_free(nmoe_tokenizer *tok);
int nmoe_tokenizer_encode(nmoe_tokenizer *tok, const char *text, uint32_t *out_ids, int max_ids);

nmoe_vocab *nmoe_vocab_load(const char *path, int quiet);
void nmoe_vocab_free(nmoe_vocab *vocab);
const char *nmoe_vocab_decode_token(const nmoe_vocab *vocab, int token_id);

#ifdef __cplusplus
}
#endif

#endif
