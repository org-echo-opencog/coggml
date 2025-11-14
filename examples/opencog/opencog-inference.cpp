#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-opencog.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

int main(int argc, char ** argv) {
    ggml_log_set(ggml_log_callback_default, nullptr);
    
    printf("OpenCog Pure Inference Engine Demo\n");
    printf("===================================\n\n");

    // Initialize ggml context
    struct ggml_init_params params = {
        /*.mem_size   =*/ 16*1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    
    // Initialize CPU backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    
    // Create AtomSpace
    printf("1. Initializing AtomSpace...\n");
    struct ggml_opencog_atomspace * atomspace = 
        ggml_opencog_atomspace_init(ctx, backend, 1000, 64);
    
    // Create some concept nodes
    printf("2. Creating concept nodes...\n");
    struct ggml_opencog_truth_value tv_high = ggml_opencog_tv_create(0.9f, 0.8f, 10.0f);
    struct ggml_opencog_truth_value tv_medium = ggml_opencog_tv_create(0.7f, 0.6f, 5.0f);
    
    struct ggml_opencog_atom * animal = ggml_opencog_atom_create(
        atomspace, OPENCOG_ATOM_CONCEPT_NODE, "Animal", tv_high);
    
    struct ggml_opencog_atom * mammal = ggml_opencog_atom_create(
        atomspace, OPENCOG_ATOM_CONCEPT_NODE, "Mammal", tv_high);
    
    struct ggml_opencog_atom * dog = ggml_opencog_atom_create(
        atomspace, OPENCOG_ATOM_CONCEPT_NODE, "Dog", tv_high);
    
    struct ggml_opencog_atom * canine = ggml_opencog_atom_create(
        atomspace, OPENCOG_ATOM_CONCEPT_NODE, "Canine", tv_medium);
    
    printf("   Created atoms: Animal, Mammal, Dog, Canine\n");
    
    // Create inheritance relationships
    printf("3. Creating inheritance relationships...\n");
    
    // Mammal -> Animal
    ggml_opencog_atom_add_link(atomspace, mammal, animal, 
                              OPENCOG_ATOM_INHERITANCE_LINK, tv_high);
    
    // Dog -> Mammal  
    ggml_opencog_atom_add_link(atomspace, dog, mammal,
                              OPENCOG_ATOM_INHERITANCE_LINK, tv_high);
    
    // Dog -> Canine
    ggml_opencog_atom_add_link(atomspace, dog, canine,
                              OPENCOG_ATOM_INHERITANCE_LINK, tv_medium);
    
    printf("   Created links: Dog->Mammal, Mammal->Animal, Dog->Canine\n");
    
    // Initialize Unified Rule Engine
    printf("4. Initializing Unified Rule Engine (URE)...\n");
    struct ggml_opencog_ure * ure = ggml_opencog_ure_init(atomspace, 10, 0.5f);
    
    // Create and add inference rules
    struct ggml_opencog_inference_rule inheritance_rule;
    inheritance_rule.name = "Inheritance Transitivity";
    inheritance_rule.precondition = ggml_opencog_rule_inheritance_precondition;
    inheritance_rule.conclusion = ggml_opencog_rule_inheritance_conclusion;
    inheritance_rule.confidence_boost = 0.1f;
    
    ggml_opencog_ure_add_rule(ure, &inheritance_rule);
    
    printf("   Added inheritance transitivity rule\n");
    
    // Perform forward chaining inference
    printf("5. Performing forward chaining inference...\n");
    int inferences = ggml_opencog_ure_forward_chain(ure, nullptr);
    
    printf("   Made %d inferences\n", inferences);
    
    // Query the knowledge base
    printf("6. Querying knowledge base...\n");
    
    // Create a pattern to find all concept nodes
    struct ggml_opencog_atom pattern;
    pattern.type = OPENCOG_ATOM_CONCEPT_NODE;
    pattern.name = nullptr;
    pattern.tv = ggml_opencog_tv_create(0.0f, 0.0f, 0.0f);
    
    int n_results = 0;
    struct ggml_opencog_atom ** results = ggml_opencog_query(atomspace, &pattern, &n_results);
    
    printf("   Found %d concept nodes with confidence >= 0.5:\n", n_results);
    for (int i = 0; i < n_results; i++) {
        printf("     - %s (strength: %.2f, confidence: %.2f)\n", 
               results[i]->name ? results[i]->name : "Anonymous",
               results[i]->tv.strength,
               results[i]->tv.confidence);
    }
    
    // Demonstrate truth value operations
    printf("7. Demonstrating truth value operations...\n");
    
    struct ggml_opencog_truth_value tv1 = ggml_opencog_tv_create(0.8f, 0.9f, 5.0f);
    struct ggml_opencog_truth_value tv2 = ggml_opencog_tv_create(0.7f, 0.8f, 3.0f);
    
    struct ggml_opencog_truth_value tv_and = ggml_opencog_tv_and(tv1, tv2);
    struct ggml_opencog_truth_value tv_or = ggml_opencog_tv_or(tv1, tv2);
    struct ggml_opencog_truth_value tv_not = ggml_opencog_tv_not(tv1);
    
    printf("   TV1: (%.2f, %.2f)\n", tv1.strength, tv1.confidence);
    printf("   TV2: (%.2f, %.2f)\n", tv2.strength, tv2.confidence);
    printf("   AND: (%.2f, %.2f)\n", tv_and.strength, tv_and.confidence);
    printf("   OR:  (%.2f, %.2f)\n", tv_or.strength, tv_or.confidence);
    printf("   NOT: (%.2f, %.2f)\n", tv_not.strength, tv_not.confidence);
    
    // Print final AtomSpace statistics
    printf("\n8. Final AtomSpace statistics:\n");
    printf("   Total atoms: %d/%d\n", atomspace->n_atoms, atomspace->capacity);
    printf("   Embedding dimension: %d\n", atomspace->embedding_dim);
    
    // Clean up
    printf("\n9. Cleaning up...\n");
    free(results);
    ggml_opencog_ure_free(ure);
    ggml_opencog_atomspace_free(atomspace);
    ggml_backend_free(backend);
    ggml_free(ctx);
    
    printf("\nOpenCog Pure Inference Engine demo completed successfully!\n");
    
    return 0;
}