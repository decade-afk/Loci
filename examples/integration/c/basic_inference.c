#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "loci.h"

int main(void) {
    const char *model = "D:/OpenProject/Qwen_Qwen3-0.6B-Q5_K_L.gguf";
    LociEngine *engine = loci_engine_new(model, 512, 0);
    if (engine == NULL) {
        const char *err = loci_get_last_error();
        fprintf(stderr, "loci_engine_new failed: %s\n", err ? err : "(no error)");
        return 1;
    }

    char *resp = loci_generate(engine, "Hello from C", 32, 0.7f);
    if (resp == NULL) {
        const char *err = loci_get_last_error();
        fprintf(stderr, "loci_generate failed: %s\n", err ? err : "(no error)");
        loci_engine_free(engine);
        return 1;
    }

    printf("response: %s\n", resp);
    loci_free_string(resp);
    loci_engine_free(engine);
    return 0;
}
