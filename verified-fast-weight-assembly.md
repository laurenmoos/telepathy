# Verified Fast-Weight Assembly

> Early design sketch. This is a proposed experiment, not yet an established method.

## Core idea

A frozen transformer produces internal activation states in J-space. We use those states to construct small, temporary transformations of the model's computation. A verifier tests the transformations, and useful ones are cached for later reuse and composition.

```text
transformer activations
        -> candidate J-space transformation
        -> temporary execution
        -> verifier
        -> retain or discard
        -> compose retained transformations
```

The transformer itself remains frozen. What grows is an external library of small programs that modify its activations or effective weights during inference.

## The pieces

### Steering is the execution mechanism

An atomic instruction changes an activation state or temporarily changes how a layer processes that state.

One simple instruction is a low-rank residual transformation:

```text
T(J) = J + U(V^T J)
```

This differs from adding one constant steering vector because the change depends on the current state `J`. It is still steering in the broad sense.

### Symbolic regression is the search analogy

The search problem is to construct a short expression from available transformations:

```text
P(J) = T3(T2(T1(J)))
```

Candidate programs are scored by their effect on the model's output, intermediate activation targets, and program cost. Early experiments can enumerate short programs rather than use a sophisticated search algorithm.

### Assembly theory supplies the reuse principle

Accepted transformations are added to an assembly pool. They remain available as parts of later programs.

```text
L_(t+1) = L_t union {new verified transformation}
```

This does not guarantee that the model becomes generally more intelligent. It gives two narrower monotonic properties:

- the best score available on the retained verifier suite cannot decrease, because previous programs remain available;
- the shortest known construction of a target state can stay the same or become shorter as reusable parts accumulate.

These guarantees disappear outside the verifier distribution. A growing library can still overfit, become difficult to search, or contain transformations that interfere when composed.

### Fast weights are the storage mechanism

The transformations can be viewed as temporary weight changes rather than permanent fine-tuning:

```text
W_effective = W_frozen + delta_W_program
```

Initially, an external procedure can derive `delta_W_program` from activation pairs. A later version could let the transformer's own hidden states generate low-rank factors directly, for example through an outer product:

```text
delta_W = value key^T
```

Only transformations that pass the verifier are cached. The frozen base model and every previous accepted program remain recoverable.

## Why this instead of Markdown?

Markdown should win whenever the model can state a useful strategy and reliably reconstruct the computation from that description.

Fast-weight assembly is interesting only when a reusable internal transformation:

- is difficult to express in language;
- preserves a perceptual or intermediate state that language discards;
- works more reliably than a textual instruction;
- or composes into useful behavior that its individual parts do not produce.

The intended domain is therefore partly pre-verbal computation: perception, spatial transformation, analogy, and other processes where the model may contain useful structure before it has a clean proposition to report.

## What would make it program-like?

A single intervention is steering. The program claim requires additional evidence:

- the same transformation works on multiple input states;
- applying it repeatedly has a predictable effect;
- order matters when different transformations are composed;
- intermediate J-states move toward predicted counterfactual states;
- reused parts outperform separately fitting a new intervention for every case.

The distinction is therefore not between programs and steering. It is between isolated steering and a reusable algebra of steering transformations.

## Hello-world experiment: a reusable latent rotation

### Model

Use **PaliGemma 2 3B**, which combines a SigLIP vision encoder with a decoder initialized from Gemma 2 2B. Plain Gemma 2 2B is text-only. PaliGemma gives access to projected image-token activations at the interface between vision and language.

### Data

Generate simple images of asymmetric arrows pointing right, up, left, or down. Vary color, size, position, background, and minor shape details.

The verifier knows the correct direction because it generated the image.

### Learn one transformation

Collect interface activations for paired images related by a 90-degree counterclockwise rotation:

```text
J(right image) -> J(up image)
J(up image)    -> J(left image)
J(left image)  -> J(down image)
J(down image)  -> J(right image)
```

Fit one small low-rank transformation `R` across all of these pairs:

```text
R(J_image) approximately equals J_rotated_image
```

This first version does not ask the transformer to invent `R`. We derive `R` mechanically from its own activations.

### Execute a tiny program

For a held-out right-pointing arrow:

```text
J0 = encode(image)
J1 = R(J0)
J2 = R(J1)
J3 = R(J2)
```

Feed each state into the unchanged language decoder using the same question: “Which direction is the arrow pointing?”

The predicted outputs are:

```text
decode(J0) -> right
decode(J1) -> up
decode(J2) -> left
decode(J3) -> down
```

We also compare each transformed state with the activation produced by the corresponding real rotated image.

### Controls

- the frozen model with no intervention;
- the real rotated image as an oracle;
- a constant mean-difference steering vector;
- a separate intervention fitted for each direction;
- a textual instruction asking the model to imagine the rotation;
- a random low-rank transformation;
- applying `R` to held-out colors, positions, and arrow shapes.

### Acceptance rule

The library initially contains only the identity transformation. Add `R` only if it improves a held-out verifier set without reducing the previously retained score.

This makes the best retained result monotonic on that fixed suite:

```text
best_score(L_(t+1)) >= best_score(L_t)
```

It is an archive guarantee, not a guarantee of generalization.

### What the experiment would show

Success would show that one activation-derived, state-dependent transformation can be reused as an instruction and composed with itself to produce predictable counterfactual perceptions.

It would not yet show autonomous self-programming, open-ended improvement, or general reasoning. Failure would also be informative: image representations may not support a shared low-rank rotation operator, repeated application may leave the model's activation manifold, or the language decoder may ignore the altered visual state.

## Next step if rotation works

Learn a second transformation `M` for reflection. Rotation and reflection do not generally commute:

```text
M(R(J)) != R(M(J))
```

If the decoded outputs follow this predicted order dependence on held-out images, the library begins to behave like a small instruction set rather than a collection of unrelated steering vectors.

Only after that should the transformer itself begin proposing candidate transformations for the verifier to test and cache.

## Closest prior work

- [Assembly theory](https://www.nature.com/articles/s41586-023-06600-9) formalizes recursive construction from retained reusable parts.
- [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) interprets activation-generated outer products as instructions that program fast weights.
- [A Modern Self-Referential Weight Matrix That Learns to Modify Itself](https://arxiv.org/abs/2202.05780) studies weight matrices that update themselves during execution.
- [Language Models Can Rewrite Their Own Weights](https://arxiv.org/abs/2611.21885) demonstrates verified self-proposed weight edits at larger model scale; its smaller models did not reliably succeed.
- [Steering CLIP's Vision Transformer with Sparse Autoencoders](https://arxiv.org/abs/2504.08729) shows that internal visual features can be causally steered.
- [PaliGemma 2](https://arxiv.org/abs/2412.03555) supplies the small open vision-language architecture proposed for the experiment.

## Working description

> Verified fast-weight assembly uses a frozen transformer's own activation geometry to construct, test, retain, and compose temporary inference-time transformations. Steering supplies the primitive operations, symbolic regression supplies the program-search framing, and assembly-style retention makes verified parts cumulatively reusable.
