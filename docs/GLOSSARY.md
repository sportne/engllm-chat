# Glossary

This glossary defines the main terms used across the `engllm-chat`
documentation.

## Action envelope

The structured response object the model must return on each step. It contains
an action such as a tool request or a final response.

## Active context

The subset of session history currently kept in the prompt for the next turn.
Older turns may be removed when token limits are reached.

## Continuation

A normal workflow outcome meaning the model needs more tool budget or more tool
rounds before it can finish the turn.

## Deterministic tool

A normal application function with fixed input and output shapes. The model can
request the tool, but the tool behavior itself is defined by Python code, not
by the model.

## Markitdown conversion

The fallback path used for some non-text documents. The file is converted to
markdown text so the rest of the chat tool layer can treat it as readable
content.

## Provider-neutral

A design choice where the rest of the application avoids depending on one
provider’s custom payload format or runtime semantics.

## Readable content

The text form of a file that the model can safely inspect. This may be direct
UTF-8 text or converted markdown for supported document types.

## Schema-first

A design choice where the provider response must match a validated structured
schema before the rest of the application trusts it.

## Tool budget

The total number of tool calls the workflow allows for one user turn.

## Tool round

One cycle where the model requests tool work, the application executes it, and
the result is fed back into the conversation.
