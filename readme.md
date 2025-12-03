# libraries
- llguidance provides and easy way to compute "token masks"
    - this helps us see which tokens are valid in a particular grammar
    - **constrained decoding**
- llama cpp w/ rust bindings provides api's to control the inference loop
    - The infererance loop is the loop which keeps on predicting the next token
    - Since we control this loop we can stop, fast-forward, and change the each inference

# process
Prompt the LLM in such way where it is responding in near human like language.
Use llguidance to fast-forward templated text 
(if it is the only option valid in the grammar then we don't need generate it, only compute it for attention).
Each time we need the LLM's inference for the next token that is a *critical point*.
Use heuristic data to massage logits at critical points to what gets the best results for the validation set.

# algorithm
```
let C be a critical point

let Cₜ be the be all valid next tokens

let Cₜₑ⁺ be positive examples for a given categor
let Cₜₑ⁻ be negative examples for a given token
```
