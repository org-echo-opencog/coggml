#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// OpenCog Atom Types
enum ggml_opencog_atom_type {
    OPENCOG_ATOM_CONCEPT_NODE = 0,
    OPENCOG_ATOM_PREDICATE_NODE,
    OPENCOG_ATOM_LINK_NODE,
    OPENCOG_ATOM_INHERITANCE_LINK,
    OPENCOG_ATOM_SIMILARITY_LINK,
    OPENCOG_ATOM_IMPLICATION_LINK,
    OPENCOG_ATOM_EVALUATION_LINK,
    OPENCOG_ATOM_TYPE_COUNT
};

// Truth Value structure for probabilistic reasoning
struct ggml_opencog_truth_value {
    float strength;     // Probability/confidence [0,1]
    float confidence;   // Amount of evidence [0,1] 
    float count;        // Supporting evidence count
};

// OpenCog Atom structure
struct ggml_opencog_atom {
    enum ggml_opencog_atom_type type;
    char * name;                                    // Atom name/identifier
    struct ggml_opencog_truth_value tv;            // Truth value
    struct ggml_tensor * embedding;                // Neural representation
    struct ggml_opencog_atom ** outgoing;          // Outgoing connections
    int n_outgoing;                                 // Number of outgoing atoms
    void * data;                                    // Additional atom data
};

// AtomSpace - the core knowledge store
struct ggml_opencog_atomspace {
    struct ggml_context * ctx;                      // ggml context
    ggml_backend_t backend;                         // computation backend
    ggml_backend_buffer_t buffer;                   // memory buffer
    
    struct ggml_opencog_atom ** atoms;              // Array of atoms
    int n_atoms;                                    // Current number of atoms
    int capacity;                                   // Maximum atoms capacity
    
    struct ggml_tensor * atom_embeddings;           // Tensor for all embeddings
    int embedding_dim;                              // Embedding dimensionality
};

// PLN Inference Rule structure
struct ggml_opencog_inference_rule {
    const char * name;
    bool (*precondition)(struct ggml_opencog_atomspace * as, 
                        struct ggml_opencog_atom ** premises, int n_premises);
    struct ggml_opencog_atom * (*conclusion)(struct ggml_opencog_atomspace * as,
                                           struct ggml_opencog_atom ** premises, int n_premises);
    float confidence_boost;
};

// Unified Rule Engine (URE) 
struct ggml_opencog_ure {
    struct ggml_opencog_atomspace * atomspace;
    struct ggml_opencog_inference_rule ** rules;
    int n_rules;
    int max_iterations;
    float min_confidence;
};

// Core AtomSpace API
GGML_API struct ggml_opencog_atomspace * ggml_opencog_atomspace_init(
    struct ggml_context * ctx,
    ggml_backend_t backend, 
    int capacity,
    int embedding_dim);

GGML_API void ggml_opencog_atomspace_free(struct ggml_opencog_atomspace * as);

GGML_API struct ggml_opencog_atom * ggml_opencog_atom_create(
    struct ggml_opencog_atomspace * as,
    enum ggml_opencog_atom_type type,
    const char * name,
    struct ggml_opencog_truth_value tv);

GGML_API struct ggml_opencog_atom * ggml_opencog_link_create(
    struct ggml_opencog_atomspace * as,
    enum ggml_opencog_atom_type type,
    struct ggml_opencog_atom ** outgoing,
    int n_outgoing,
    struct ggml_opencog_truth_value tv);

GGML_API struct ggml_opencog_atom * ggml_opencog_atom_get(
    struct ggml_opencog_atomspace * as,
    const char * name);

GGML_API bool ggml_opencog_atom_add_link(
    struct ggml_opencog_atomspace * as,
    struct ggml_opencog_atom * from,
    struct ggml_opencog_atom * to,
    enum ggml_opencog_atom_type link_type,
    struct ggml_opencog_truth_value tv);

// Truth Value operations
GGML_API struct ggml_opencog_truth_value ggml_opencog_tv_create(
    float strength, float confidence, float count);

GGML_API struct ggml_opencog_truth_value ggml_opencog_tv_and(
    struct ggml_opencog_truth_value tv1,
    struct ggml_opencog_truth_value tv2);

GGML_API struct ggml_opencog_truth_value ggml_opencog_tv_or(
    struct ggml_opencog_truth_value tv1,
    struct ggml_opencog_truth_value tv2);

GGML_API struct ggml_opencog_truth_value ggml_opencog_tv_not(
    struct ggml_opencog_truth_value tv);

// PLN Inference Rules
GGML_API bool ggml_opencog_rule_modus_ponens_precondition(
    struct ggml_opencog_atomspace * as,
    struct ggml_opencog_atom ** premises, int n_premises);

GGML_API struct ggml_opencog_atom * ggml_opencog_rule_modus_ponens_conclusion(
    struct ggml_opencog_atomspace * as,
    struct ggml_opencog_atom ** premises, int n_premises);

GGML_API bool ggml_opencog_rule_inheritance_precondition(
    struct ggml_opencog_atomspace * as,
    struct ggml_opencog_atom ** premises, int n_premises);

GGML_API struct ggml_opencog_atom * ggml_opencog_rule_inheritance_conclusion(
    struct ggml_opencog_atomspace * as,
    struct ggml_opencog_atom ** premises, int n_premises);

// URE API  
GGML_API struct ggml_opencog_ure * ggml_opencog_ure_init(
    struct ggml_opencog_atomspace * as,
    int max_iterations,
    float min_confidence);

GGML_API void ggml_opencog_ure_free(struct ggml_opencog_ure * ure);

GGML_API void ggml_opencog_ure_add_rule(
    struct ggml_opencog_ure * ure,
    struct ggml_opencog_inference_rule * rule);

GGML_API int ggml_opencog_ure_forward_chain(
    struct ggml_opencog_ure * ure,
    struct ggml_opencog_atom * target);

GGML_API int ggml_opencog_ure_backward_chain(
    struct ggml_opencog_ure * ure,
    struct ggml_opencog_atom * query);

// Pure Inference Engine API
GGML_API int ggml_opencog_inference_step(
    struct ggml_opencog_ure * ure);

GGML_API struct ggml_opencog_atom ** ggml_opencog_query(
    struct ggml_opencog_atomspace * as,
    struct ggml_opencog_atom * pattern,
    int * n_results);

#ifdef __cplusplus
}
#endif