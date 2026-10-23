// QIGen-inspired schedule selection for existing exact PTQ1 kernels.
// Not QIGen's GPTQ format or a generated/retuned weight representation.
#ifndef GGML_PTQ1_QIGEN_H
#define GGML_PTQ1_QIGEN_H
#include <stdlib.h>
#include <string.h>
static int ggml_ptq1_qigen_mode(int columns) {
    const char * enabled=getenv("GGML_PTQ1_QIGEN");
    if (!enabled || strcmp(enabled,"1") || columns<1 || columns>5) return 0;
    const char * schedule=getenv("GGML_PTQ1_QIGEN_SCHEDULE");
    // Exactly five choices, indexed by actual chunk width. Invalid settings
    // fail closed to the pre-existing compile-time dispatch, never tune online.
    if (!schedule || strlen(schedule)!=5) return 0;
    for (int i=0;i<5;++i) if (schedule[i]<'0' || schedule[i]>'4') return 0;
    return schedule[columns-1]-'0';
}
#endif
