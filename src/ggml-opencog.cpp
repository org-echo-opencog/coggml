#include "ggml-opencog.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <cstring>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <string>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// Helper function to calculate truth value from strength and confidence
static float truth_value_to_probability(struct ggml_opencog_truth_value tv) {
    // Use standard OpenCog truth value semantics
    return tv.strength * tv.confidence + (1.0f - tv.confidence) * 0.5f;
}

// Helper function to create default truth value
static struct ggml_opencog_truth_value default_tv() {
    return {0.8f, 0.9f, 1.0f}; // Default reasonable values
}

//
// AtomSpace implementation
//

struct ggml_opencog_atomspace * ggml_opencog_atomspace_init(
        struct ggml_context * ctx,
        ggml_backend_t backend, 
        int capacity,
        int embedding_dim) {
    
    struct ggml_opencog_atomspace * as = (struct ggml_opencog_atomspace *)
        malloc(sizeof(struct ggml_opencog_atomspace));
    
    as->ctx = ctx;
    as->backend = backend;
    as->capacity = capacity;
    as->n_atoms = 0;
    as->embedding_dim = embedding_dim;
    
    // Allocate atoms array
    as->atoms = (struct ggml_opencog_atom **)
        calloc(capacity, sizeof(struct ggml_opencog_atom *));
    
    // Create tensor for all atom embeddings
    as->atom_embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 
                                             embedding_dim, capacity);
    ggml_set_name(as->atom_embeddings, "opencog_embeddings");
    
    // Allocate backend buffer
    size_t buffer_size = ggml_tensor_overhead() + ggml_nbytes(as->atom_embeddings);
    as->buffer = ggml_backend_alloc_buffer(backend, buffer_size);
    
    // Initialize embeddings to random values
    if (as->atom_embeddings->data) {
        float * data = (float *)as->atom_embeddings->data;
        for (int i = 0; i < capacity * embedding_dim; i++) {
            data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // [-1, 1]
        }
    }
    
    return as;
}

void ggml_opencog_atomspace_free(struct ggml_opencog_atomspace * as) {
    if (!as) return;
    
    // Free all atoms
    for (int i = 0; i < as->n_atoms; i++) {
        if (as->atoms[i]) {
            free(as->atoms[i]->name);
            free(as->atoms[i]->outgoing);
            free(as->atoms[i]);
        }
    }
    
    free(as->atoms);
    ggml_backend_buffer_free(as->buffer);
    free(as);
}

struct ggml_opencog_atom * ggml_opencog_atom_create(
        struct ggml_opencog_atomspace * as,
        enum ggml_opencog_atom_type type,
        const char * name,
        struct ggml_opencog_truth_value tv) {
    
    if (as->n_atoms >= as->capacity) {
        return nullptr;  // AtomSpace full
    }
    
    struct ggml_opencog_atom * atom = (struct ggml_opencog_atom *)
        malloc(sizeof(struct ggml_opencog_atom));
    
    atom->type = type;
    atom->name = strdup(name);
    atom->tv = tv;
    atom->outgoing = nullptr;
    atom->n_outgoing = 0;
    atom->data = nullptr;
    
    // Create embedding tensor for this atom
    atom->embedding = ggml_view_1d(as->ctx, as->atom_embeddings,
                                  as->embedding_dim,
                                  as->n_atoms * as->embedding_dim * sizeof(float));
    
    as->atoms[as->n_atoms] = atom;
    as->n_atoms++;
    
    return atom;
}

struct ggml_opencog_atom * ggml_opencog_link_create(
        struct ggml_opencog_atomspace * as,
        enum ggml_opencog_atom_type type,
        struct ggml_opencog_atom ** outgoing,
        int n_outgoing,
        struct ggml_opencog_truth_value tv) {
    
    if (as->n_atoms >= as->capacity) {
        return nullptr;
    }
    
    struct ggml_opencog_atom * link = (struct ggml_opencog_atom *)
        malloc(sizeof(struct ggml_opencog_atom));
    
    link->type = type;
    link->name = nullptr; // Links typically don't have names
    link->tv = tv;
    link->n_outgoing = n_outgoing;
    link->data = nullptr;
    
    // Copy outgoing atoms
    link->outgoing = (struct ggml_opencog_atom **)
        malloc(n_outgoing * sizeof(struct ggml_opencog_atom *));
    for (int i = 0; i < n_outgoing; i++) {
        link->outgoing[i] = outgoing[i];
    }
    
    // Create embedding for link (average of outgoing atoms)
    link->embedding = ggml_view_1d(as->ctx, as->atom_embeddings,
                                  as->embedding_dim,
                                  as->n_atoms * as->embedding_dim * sizeof(float));
    
    as->atoms[as->n_atoms] = link;
    as->n_atoms++;
    
    return link;
}

struct ggml_opencog_atom * ggml_opencog_atom_get(
        struct ggml_opencog_atomspace * as,
        const char * name) {
    
    for (int i = 0; i < as->n_atoms; i++) {
        if (as->atoms[i]->name && strcmp(as->atoms[i]->name, name) == 0) {
            return as->atoms[i];
        }
    }
    return nullptr;
}

bool ggml_opencog_atom_add_link(
        struct ggml_opencog_atomspace * as,
        struct ggml_opencog_atom * from,
        struct ggml_opencog_atom * to,
        enum ggml_opencog_atom_type link_type,
        struct ggml_opencog_truth_value tv) {
    
    struct ggml_opencog_atom * outgoing[2] = {from, to};
    struct ggml_opencog_atom * link = ggml_opencog_link_create(as, link_type, 
                                                              outgoing, 2, tv);
    return link != nullptr;
}

//
// Truth Value operations
//

struct ggml_opencog_truth_value ggml_opencog_tv_create(
        float strength, float confidence, float count) {
    struct ggml_opencog_truth_value tv;
    tv.strength = fmaxf(0.0f, fminf(1.0f, strength));
    tv.confidence = fmaxf(0.0f, fminf(1.0f, confidence)); 
    tv.count = fmaxf(0.0f, count);
    return tv;
}

struct ggml_opencog_truth_value ggml_opencog_tv_and(
        struct ggml_opencog_truth_value tv1,
        struct ggml_opencog_truth_value tv2) {
    
    // PLN conjunction rule
    float s1 = tv1.strength;
    float s2 = tv2.strength;
    float c1 = tv1.confidence;
    float c2 = tv2.confidence;
    
    float strength = s1 * s2;
    float confidence = c1 * c2;
    float count = tv1.count + tv2.count;
    
    return ggml_opencog_tv_create(strength, confidence, count);
}

struct ggml_opencog_truth_value ggml_opencog_tv_or(
        struct ggml_opencog_truth_value tv1,
        struct ggml_opencog_truth_value tv2) {
    
    // PLN disjunction rule
    float s1 = tv1.strength;
    float s2 = tv2.strength;
    float c1 = tv1.confidence;
    float c2 = tv2.confidence;
    
    float strength = s1 + s2 - s1 * s2;
    float confidence = fminf(c1, c2);
    float count = fmaxf(tv1.count, tv2.count);
    
    return ggml_opencog_tv_create(strength, confidence, count);
}

struct ggml_opencog_truth_value ggml_opencog_tv_not(
        struct ggml_opencog_truth_value tv) {
    
    float strength = 1.0f - tv.strength;
    float confidence = tv.confidence;
    float count = tv.count;
    
    return ggml_opencog_tv_create(strength, confidence, count);
}

//
// PLN Inference Rules
//

bool ggml_opencog_rule_modus_ponens_precondition(
        struct ggml_opencog_atomspace * as,
        struct ggml_opencog_atom ** premises, int n_premises) {
    
    // Modus Ponens requires: P, P->Q
    if (n_premises != 2) return false;
    
    struct ggml_opencog_atom * p = premises[0];
    struct ggml_opencog_atom * implication = premises[1];
    
    // Check if second premise is an implication
    if (implication->type != OPENCOG_ATOM_IMPLICATION_LINK) return false;
    if (implication->n_outgoing != 2) return false;
    
    // Check if first part of implication matches P
    return implication->outgoing[0] == p;
}

struct ggml_opencog_atom * ggml_opencog_rule_modus_ponens_conclusion(
        struct ggml_opencog_atomspace * as,
        struct ggml_opencog_atom ** premises, int n_premises) {
    
    if (!ggml_opencog_rule_modus_ponens_precondition(as, premises, n_premises)) {
        return nullptr;
    }
    
    struct ggml_opencog_atom * p = premises[0];
    struct ggml_opencog_atom * implication = premises[1];
    struct ggml_opencog_atom * q = implication->outgoing[1];
    
    // Calculate new truth value using PLN deduction rule
    struct ggml_opencog_truth_value tv_p = p->tv;
    struct ggml_opencog_truth_value tv_imp = implication->tv;
    
    float strength = tv_p.strength * tv_imp.strength;
    float confidence = tv_p.confidence * tv_imp.confidence;
    float count = fminf(tv_p.count, tv_imp.count);
    
    struct ggml_opencog_truth_value new_tv = ggml_opencog_tv_create(strength, confidence, count);
    
    // Update Q's truth value or create new atom if needed
    q->tv = ggml_opencog_tv_or(q->tv, new_tv);
    
    return q;
}

bool ggml_opencog_rule_inheritance_precondition(
        struct ggml_opencog_atomspace * as,
        struct ggml_opencog_atom ** premises, int n_premises) {
    
    // Inheritance transitivity: A->B, B->C
    if (n_premises != 2) return false;
    
    struct ggml_opencog_atom * inh1 = premises[0];
    struct ggml_opencog_atom * inh2 = premises[1];
    
    if (inh1->type != OPENCOG_ATOM_INHERITANCE_LINK) return false;
    if (inh2->type != OPENCOG_ATOM_INHERITANCE_LINK) return false;
    if (inh1->n_outgoing != 2 || inh2->n_outgoing != 2) return false;
    
    // Check if B matches (end of first, start of second)
    return inh1->outgoing[1] == inh2->outgoing[0];
}

struct ggml_opencog_atom * ggml_opencog_rule_inheritance_conclusion(
        struct ggml_opencog_atomspace * as,
        struct ggml_opencog_atom ** premises, int n_premises) {
    
    if (!ggml_opencog_rule_inheritance_precondition(as, premises, n_premises)) {
        return nullptr;
    }
    
    struct ggml_opencog_atom * inh1 = premises[0]; // A->B
    struct ggml_opencog_atom * inh2 = premises[1]; // B->C
    
    struct ggml_opencog_atom * a = inh1->outgoing[0];
    struct ggml_opencog_atom * c = inh2->outgoing[1];
    
    // Calculate transitivity strength
    float strength = inh1->tv.strength * inh2->tv.strength;
    float confidence = inh1->tv.confidence * inh2->tv.confidence * 0.9f; // Slight decay
    float count = fminf(inh1->tv.count, inh2->tv.count);
    
    struct ggml_opencog_truth_value new_tv = ggml_opencog_tv_create(strength, confidence, count);
    
    // Create A->C inheritance link
    struct ggml_opencog_atom * outgoing[2] = {a, c};
    return ggml_opencog_link_create(as, OPENCOG_ATOM_INHERITANCE_LINK, outgoing, 2, new_tv);
}

//
// Unified Rule Engine (URE)
//

struct ggml_opencog_ure * ggml_opencog_ure_init(
        struct ggml_opencog_atomspace * as,
        int max_iterations,
        float min_confidence) {
    
    struct ggml_opencog_ure * ure = (struct ggml_opencog_ure *)
        malloc(sizeof(struct ggml_opencog_ure));
    
    ure->atomspace = as;
    ure->max_iterations = max_iterations;
    ure->min_confidence = min_confidence;
    ure->rules = nullptr;
    ure->n_rules = 0;
    
    return ure;
}

void ggml_opencog_ure_free(struct ggml_opencog_ure * ure) {
    if (!ure) return;
    
    free(ure->rules);
    free(ure);
}

void ggml_opencog_ure_add_rule(
        struct ggml_opencog_ure * ure,
        struct ggml_opencog_inference_rule * rule) {
    
    ure->rules = (struct ggml_opencog_inference_rule **)
        realloc(ure->rules, (ure->n_rules + 1) * sizeof(struct ggml_opencog_inference_rule *));
    ure->rules[ure->n_rules] = rule;
    ure->n_rules++;
}

int ggml_opencog_ure_forward_chain(
        struct ggml_opencog_ure * ure,
        struct ggml_opencog_atom * target) {
    
    int inferences_made = 0;
    
    for (int iter = 0; iter < ure->max_iterations; iter++) {
        bool made_inference = false;
        
        // Try each inference rule
        for (int rule_idx = 0; rule_idx < ure->n_rules; rule_idx++) {
            struct ggml_opencog_inference_rule * rule = ure->rules[rule_idx];
            
            // Try all possible combinations of premises
            for (int i = 0; i < ure->atomspace->n_atoms; i++) {
                for (int j = i + 1; j < ure->atomspace->n_atoms; j++) {
                    struct ggml_opencog_atom * premises[2] = {
                        ure->atomspace->atoms[i],
                        ure->atomspace->atoms[j]
                    };
                    
                    if (rule->precondition(ure->atomspace, premises, 2)) {
                        struct ggml_opencog_atom * conclusion = 
                            rule->conclusion(ure->atomspace, premises, 2);
                        
                        if (conclusion && conclusion->tv.confidence >= ure->min_confidence) {
                            made_inference = true;
                            inferences_made++;
                            
                            // Check if we reached target
                            if (conclusion == target) {
                                return inferences_made;
                            }
                        }
                    }
                }
            }
        }
        
        if (!made_inference) break; // No more inferences possible
    }
    
    return inferences_made;
}

int ggml_opencog_ure_backward_chain(
        struct ggml_opencog_ure * ure,
        struct ggml_opencog_atom * query) {
    
    // Simplified backward chaining implementation
    // In practice, this would use goal-directed search
    return ggml_opencog_ure_forward_chain(ure, query);
}

//
// Pure Inference Engine API
//

int ggml_opencog_inference_step(struct ggml_opencog_ure * ure) {
    return ggml_opencog_ure_forward_chain(ure, nullptr);
}

struct ggml_opencog_atom ** ggml_opencog_query(
        struct ggml_opencog_atomspace * as,
        struct ggml_opencog_atom * pattern,
        int * n_results) {
    
    // Simple pattern matching implementation
    std::vector<struct ggml_opencog_atom *> results;
    
    for (int i = 0; i < as->n_atoms; i++) {
        struct ggml_opencog_atom * atom = as->atoms[i];
        
        // Match by type (simple pattern matching)
        if (atom->type == pattern->type && atom->tv.confidence >= 0.5f) {
            results.push_back(atom);
        }
    }
    
    *n_results = results.size();
    if (results.empty()) return nullptr;
    
    struct ggml_opencog_atom ** result_array = (struct ggml_opencog_atom **)
        malloc(results.size() * sizeof(struct ggml_opencog_atom *));
    
    for (size_t i = 0; i < results.size(); i++) {
        result_array[i] = results[i];
    }
    
    return result_array;
}