# OpenCog Pure Inference Engine

This directory contains an implementation of OpenCog as a pure inference engine built on top of the ggml tensor library.

## Overview

The OpenCog inference engine provides:

- **AtomSpace**: A hypergraph knowledge store for representing atoms and their relationships
- **Truth Values**: Probabilistic logic values with strength, confidence, and count
- **Inference Rules**: PLN (Probabilistic Logic Networks) inference rules for automated reasoning
- **URE**: Unified Rule Engine for forward and backward chaining
- **Neural Integration**: Tensor-based embeddings for symbolic-connectionist integration

## Architecture

### Core Components

1. **AtomSpace (`ggml_opencog_atomspace`)**
   - Stores atoms (nodes and links) in a hypergraph structure
   - Backed by ggml tensors for efficient neural representation
   - Supports various atom types: ConceptNode, PredicateNode, InheritanceLink, etc.

2. **Atoms (`ggml_opencog_atom`)**
   - Fundamental units of knowledge representation
   - Each atom has a type, name, truth value, and optional neural embedding
   - Links can connect multiple atoms to represent relationships

3. **Truth Values (`ggml_opencog_truth_value`)**
   - Represent uncertain knowledge with strength [0,1] and confidence [0,1]
   - Support logical operations: AND, OR, NOT
   - Enable probabilistic reasoning over uncertain facts

4. **Inference Rules (`ggml_opencog_inference_rule`)**
   - Implement PLN inference patterns
   - Include modus ponens, inheritance transitivity, etc.
   - Can be added dynamically to the URE

5. **URE (`ggml_opencog_ure`)**
   - Manages and applies inference rules
   - Supports forward and backward chaining
   - Controls inference iterations and confidence thresholds

## Usage

### Basic Example

```cpp
#include "ggml-opencog.h"
#include "ggml-cpu.h"

// Initialize ggml context and backend
struct ggml_context * ctx = ggml_init(params);
ggml_backend_t backend = ggml_backend_cpu_init();

// Create AtomSpace
struct ggml_opencog_atomspace * atomspace = 
    ggml_opencog_atomspace_init(ctx, backend, 1000, 64);

// Create atoms
struct ggml_opencog_truth_value tv = ggml_opencog_tv_create(0.9f, 0.8f, 10.0f);
struct ggml_opencog_atom * dog = ggml_opencog_atom_create(
    atomspace, OPENCOG_ATOM_CONCEPT_NODE, "Dog", tv);

// Create relationships
ggml_opencog_atom_add_link(atomspace, dog, animal, 
    OPENCOG_ATOM_INHERITANCE_LINK, tv);

// Setup inference
struct ggml_opencog_ure * ure = ggml_opencog_ure_init(atomspace, 10, 0.5f);

// Add inference rules
struct ggml_opencog_inference_rule inheritance_rule;
inheritance_rule.name = "Inheritance Transitivity";
inheritance_rule.precondition = ggml_opencog_rule_inheritance_precondition;
inheritance_rule.conclusion = ggml_opencog_rule_inheritance_conclusion;
ggml_opencog_ure_add_rule(ure, &inheritance_rule);

// Perform inference
int inferences = ggml_opencog_ure_forward_chain(ure, nullptr);

// Query results
int n_results = 0;
struct ggml_opencog_atom ** results = ggml_opencog_query(atomspace, &pattern, &n_results);

// Cleanup
ggml_opencog_atomspace_free(atomspace);
ggml_backend_free(backend);
ggml_free(ctx);
```

### Truth Value Operations

```cpp
// Create truth values
struct ggml_opencog_truth_value tv1 = ggml_opencog_tv_create(0.8f, 0.9f, 5.0f);
struct ggml_opencog_truth_value tv2 = ggml_opencog_tv_create(0.7f, 0.8f, 3.0f);

// Logical operations
struct ggml_opencog_truth_value tv_and = ggml_opencog_tv_and(tv1, tv2);
struct ggml_opencog_truth_value tv_or = ggml_opencog_tv_or(tv1, tv2);  
struct ggml_opencog_truth_value tv_not = ggml_opencog_tv_not(tv1);
```

## Building

The OpenCog inference engine is built as part of the main ggml project:

```bash
cd coggml/build
make opencog-inference
./bin/opencog-inference
```

Run tests:
```bash
make test-opencog
./bin/test-opencog
```

## Features

### Implemented

- ✅ AtomSpace with tensor-backed storage
- ✅ Basic atom types (ConceptNode, PredicateNode, InheritanceLink, etc.)
- ✅ Truth value system with probabilistic logic operations
- ✅ PLN inference rules (inheritance transitivity, modus ponens)
- ✅ Unified Rule Engine with forward chaining
- ✅ Pattern matching and querying
- ✅ Neural embeddings for atoms
- ✅ Comprehensive test suite

### Planned Extensions

- 🔄 Backward chaining inference
- 🔄 Additional PLN inference rules (deduction, induction, abduction)
- 🔄 Attention allocation mechanisms
- 🔄 Learning and adaptation capabilities
- 🔄 Scheme/Guile integration layer
- 🔄 Distributed inference across multiple backends

## Performance

The implementation leverages ggml's optimized tensor operations:

- **Memory Efficient**: Atoms share tensor storage for embeddings
- **Backend Agnostic**: Runs on CPU, GPU (CUDA, Metal, OpenCL)
- **Scalable**: Designed to handle thousands of atoms efficiently
- **Concurrent**: Thread-safe operations where applicable

## Integration with ggml

The OpenCog inference engine seamlessly integrates with ggml's ecosystem:

- Uses ggml contexts for memory management
- Leverages ggml backends (CPU, CUDA, Metal, etc.)
- Shares tensor operations with neural networks
- Compatible with existing ggml-based applications

## API Reference

See `include/ggml-opencog.h` for the complete API documentation.

## Examples

- `opencog-inference.cpp`: Basic demonstration of inference capabilities
- `../tests/test-opencog.cpp`: Comprehensive test suite showing all features

## License

This implementation follows the same license as the parent ggml project.