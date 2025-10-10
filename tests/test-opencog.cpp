#include "ggml-opencog.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cassert>

static void test_atomspace_creation() {
    printf("Testing AtomSpace creation...\n");
    
    // Initialize ggml context
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    ggml_backend_t backend = ggml_backend_cpu_init();
    
    struct ggml_opencog_atomspace * as = ggml_opencog_atomspace_init(ctx, backend, 100, 32);
    assert(as != nullptr);
    assert(as->capacity == 100);
    assert(as->embedding_dim == 32);
    assert(as->n_atoms == 0);
    
    ggml_opencog_atomspace_free(as);
    ggml_backend_free(backend);
    ggml_free(ctx);
    
    printf("  ✓ AtomSpace creation test passed\n");
}

static void test_atom_creation() {
    printf("Testing Atom creation...\n");
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    ggml_backend_t backend = ggml_backend_cpu_init();
    
    struct ggml_opencog_atomspace * as = ggml_opencog_atomspace_init(ctx, backend, 100, 32);
    
    struct ggml_opencog_truth_value tv = ggml_opencog_tv_create(0.8f, 0.9f, 5.0f);
    
    struct ggml_opencog_atom * atom = ggml_opencog_atom_create(
        as, OPENCOG_ATOM_CONCEPT_NODE, "TestConcept", tv);
    
    assert(atom != nullptr);
    assert(atom->type == OPENCOG_ATOM_CONCEPT_NODE);
    assert(strcmp(atom->name, "TestConcept") == 0);
    assert(atom->tv.strength == 0.8f);
    assert(atom->tv.confidence == 0.9f);
    assert(as->n_atoms == 1);
    
    // Test atom retrieval
    struct ggml_opencog_atom * retrieved = ggml_opencog_atom_get(as, "TestConcept");
    assert(retrieved == atom);
    
    ggml_opencog_atomspace_free(as);
    ggml_backend_free(backend);
    ggml_free(ctx);
    
    printf("  ✓ Atom creation test passed\n");
}

static void test_truth_values() {
    printf("Testing Truth Value operations...\n");
    
    struct ggml_opencog_truth_value tv1 = ggml_opencog_tv_create(0.8f, 0.9f, 5.0f);
    struct ggml_opencog_truth_value tv2 = ggml_opencog_tv_create(0.6f, 0.7f, 3.0f);
    
    // Test AND operation
    struct ggml_opencog_truth_value tv_and = ggml_opencog_tv_and(tv1, tv2);
    assert(tv_and.strength == 0.8f * 0.6f); // 0.48
    assert(tv_and.confidence == 0.9f * 0.7f); // 0.63
    
    // Test OR operation  
    struct ggml_opencog_truth_value tv_or = ggml_opencog_tv_or(tv1, tv2);
    assert(tv_or.strength == 0.8f + 0.6f - 0.8f * 0.6f); // 0.92
    
    // Test NOT operation
    struct ggml_opencog_truth_value tv_not = ggml_opencog_tv_not(tv1);
    assert(tv_not.strength == 1.0f - 0.8f); // 0.2
    assert(tv_not.confidence == 0.9f);
    
    // Test bounds checking
    struct ggml_opencog_truth_value tv_bounds = ggml_opencog_tv_create(1.5f, -0.5f, -1.0f);
    assert(tv_bounds.strength == 1.0f); // Clamped to [0,1]
    assert(tv_bounds.confidence == 0.0f); // Clamped to [0,1]
    assert(tv_bounds.count == 0.0f); // Clamped to >=0
    
    printf("  ✓ Truth Value operations test passed\n");
}

static void test_inheritance_inference() {
    printf("Testing Inheritance inference...\n");
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    ggml_backend_t backend = ggml_backend_cpu_init();
    
    struct ggml_opencog_atomspace * as = ggml_opencog_atomspace_init(ctx, backend, 100, 32);
    
    // Create atoms
    struct ggml_opencog_truth_value tv_high = ggml_opencog_tv_create(0.9f, 0.8f, 10.0f);
    
    struct ggml_opencog_atom * a = ggml_opencog_atom_create(as, OPENCOG_ATOM_CONCEPT_NODE, "A", tv_high);
    struct ggml_opencog_atom * b = ggml_opencog_atom_create(as, OPENCOG_ATOM_CONCEPT_NODE, "B", tv_high);
    struct ggml_opencog_atom * c = ggml_opencog_atom_create(as, OPENCOG_ATOM_CONCEPT_NODE, "C", tv_high);
    
    // Create inheritance links A->B, B->C
    struct ggml_opencog_atom * outgoing1[2] = {a, b};
    struct ggml_opencog_atom * link1 = ggml_opencog_link_create(
        as, OPENCOG_ATOM_INHERITANCE_LINK, outgoing1, 2, tv_high);
    
    struct ggml_opencog_atom * outgoing2[2] = {b, c};
    struct ggml_opencog_atom * link2 = ggml_opencog_link_create(
        as, OPENCOG_ATOM_INHERITANCE_LINK, outgoing2, 2, tv_high);
    
    // Test inheritance rule precondition
    struct ggml_opencog_atom * premises[2] = {link1, link2};
    bool precondition_met = ggml_opencog_rule_inheritance_precondition(as, premises, 2);
    assert(precondition_met);
    
    // Test inference conclusion (should create A->C)
    struct ggml_opencog_atom * conclusion = ggml_opencog_rule_inheritance_conclusion(as, premises, 2);
    assert(conclusion != nullptr);
    assert(conclusion->type == OPENCOG_ATOM_INHERITANCE_LINK);
    assert(conclusion->n_outgoing == 2);
    assert(conclusion->outgoing[0] == a);
    assert(conclusion->outgoing[1] == c);
    
    printf("  ✓ Inheritance inference test passed\n");
    
    ggml_opencog_atomspace_free(as);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

static void test_ure_functionality() {
    printf("Testing URE functionality...\n");
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    ggml_backend_t backend = ggml_backend_cpu_init();
    
    struct ggml_opencog_atomspace * as = ggml_opencog_atomspace_init(ctx, backend, 100, 32);
    struct ggml_opencog_ure * ure = ggml_opencog_ure_init(as, 5, 0.5f);
    
    assert(ure != nullptr);
    assert(ure->atomspace == as);
    assert(ure->max_iterations == 5);
    assert(ure->min_confidence == 0.5f);
    assert(ure->n_rules == 0);
    
    // Add a rule
    struct ggml_opencog_inference_rule rule;
    rule.name = "Test Rule";
    rule.precondition = ggml_opencog_rule_inheritance_precondition;
    rule.conclusion = ggml_opencog_rule_inheritance_conclusion;
    rule.confidence_boost = 0.1f;
    
    ggml_opencog_ure_add_rule(ure, &rule);
    assert(ure->n_rules == 1);
    
    printf("  ✓ URE functionality test passed\n");
    
    ggml_opencog_ure_free(ure);
    ggml_opencog_atomspace_free(as);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

static void test_pattern_matching() {
    printf("Testing pattern matching...\n");
    
    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    
    struct ggml_context * ctx = ggml_init(params);
    ggml_backend_t backend = ggml_backend_cpu_init();
    
    struct ggml_opencog_atomspace * as = ggml_opencog_atomspace_init(ctx, backend, 100, 32);
    
    // Create some atoms
    struct ggml_opencog_truth_value tv_high = ggml_opencog_tv_create(0.9f, 0.8f, 10.0f);
    struct ggml_opencog_truth_value tv_low = ggml_opencog_tv_create(0.3f, 0.4f, 1.0f);
    
    ggml_opencog_atom_create(as, OPENCOG_ATOM_CONCEPT_NODE, "Concept1", tv_high);
    ggml_opencog_atom_create(as, OPENCOG_ATOM_CONCEPT_NODE, "Concept2", tv_high);
    ggml_opencog_atom_create(as, OPENCOG_ATOM_CONCEPT_NODE, "Concept3", tv_low);
    ggml_opencog_atom_create(as, OPENCOG_ATOM_PREDICATE_NODE, "Predicate1", tv_high);
    
    // Query for concept nodes  
    struct ggml_opencog_atom pattern;
    pattern.type = OPENCOG_ATOM_CONCEPT_NODE;
    pattern.name = nullptr;
    pattern.tv = ggml_opencog_tv_create(0.0f, 0.0f, 0.0f);
    
    int n_results = 0;
    struct ggml_opencog_atom ** results = ggml_opencog_query(as, &pattern, &n_results);
    
    // Should find 2 concept nodes with confidence >= 0.5 (Concept1, Concept2)
    assert(n_results == 2);
    
    free(results);
    
    printf("  ✓ Pattern matching test passed\n");
    
    ggml_opencog_atomspace_free(as);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

int main() {
    printf("OpenCog Inference Engine Test Suite\n");
    printf("====================================\n\n");
    
    test_atomspace_creation();
    test_atom_creation();
    test_truth_values();
    test_inheritance_inference();
    test_ure_functionality();
    test_pattern_matching();
    
    printf("\n✓ All OpenCog tests passed successfully!\n");
    
    return 0;
}