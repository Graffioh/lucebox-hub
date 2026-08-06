# DiffPin

**Making agent prompts prefix-cacheable by floating volatility out of the stable head**

---

## In one sentence

**DiffPin** finds the small changing part of a long prompt (a session clock, a banner, a one-line header), moves it *after* everything that stayed the same, and tells the inference engine: *pin the KV cache here — this contiguous block will come back*.

---

## Why this exists

Modern agents send a huge head on every turn: identity, rules, dozens of tool schemas. That head is expensive to **prefill** — the model must write a **KV cache** (the memory of “what I already saw”) before it can answer.

Most of that head does not change. A few tokens do — often a line like *Conversation started: Friday, July 31…*.

Prefix caching only works when the **beginning** of the prompt matches exactly. One changed date in the middle of the head, and the engine treats the whole head as new. You pay the long prefill again.

```mermaid
flowchart LR
  subgraph cold ["Without DiffPin"]
    A1["Same tools + identity"]
    A2["Different clock"]
    A3["Full head looks new"]
    A4["Rebuild all KV 💸"]
    A1 --> A2 --> A3 --> A4
  end
```

The waste is not “bad networking.” It is **throwing away valid KV** because a tiny volatile island sat where the pin key could not ignore it.

---

## The invention: DiffPin

**Name:** DiffPin  
**Idea:** Treat volatility as a *diff hunk*, not as fate.

Compare today’s tools/system head to recent traffic. The shared parts form a **prefix** and a **suffix**. The disagreement is a small **middle** (the clock). DiffPin rewrites the head so stable tokens become one contiguous block, with the volatile middle after them:

```text
Before:  [ stable … TIME … stable … ]
After:   [ stable ……… stable ][ TIME ][ end ]
              ↑ pin here
```

```mermaid
flowchart TB
  subgraph before ["What the client sent"]
    B1["████████"]
    B2["🕐 clock"]
    B3["████"]
    B1 --- B2 --- B3
  end

  subgraph after ["What DiffPin serves"]
    A1["████████████"]
    A2["🕐 clock"]
    A1 --- A2
  end

  before -->|"diff → float the clock"| after
  A1 -.->|"protected pin<br/>reuse this KV"| KV["KV cache"]
```

Same information. Better *shape* for caching.

---

## How to picture it

Think of a highlighter and a sticky note.

1. **Diff** — lay two prompts side by side; highlight what matches from the left and from the right; the unhighlighted island is the sticky note (time, model line, …).
2. **Float** — peel the sticky note off the middle and place it after the highlighted paper.
3. **Pin** — stack the highlighted paper in the KV closet. Next time the sticky note says a different day, the paper still matches.

```mermaid
sequenceDiagram
  participant Turn1 as Turn with July 30
  participant DiffPin
  participant Turn2 as Turn with July 31
  participant KV as KV pin

  Turn1->>DiffPin: Long head + clock A
  DiffPin->>KV: Remember contiguous stable head
  Turn2->>DiffPin: Same head + clock B
  DiffPin->>DiffPin: Diff spots only the clock
  DiffPin->>DiffPin: Float clock after stable block
  DiffPin->>KV: Stable block matches — restore
  Note over Turn2,KV: Prefill only clock + new user text
```

---

## What “pin-friendly” means

A layout is **pin-friendly** when the tokens you want to reuse form one uninterrupted prefix:

```mermaid
flowchart LR
  subgraph friendly ["Pin-friendly"]
    S["STABLE STABLE STABLE"]
    V["volatile"]
    T["transcript…"]
    S --> V --> T
  end
```

Not pin-friendly when volatility punches a hole in the middle — the shared suffix cannot join the pin, because prefix caches only care about *beginnings*:

```mermaid
flowchart LR
  subgraph unfriendly ["Not pin-friendly"]
    S1["STABLE"]
    V["volatile"]
    S2["STABLE"]
    T["transcript…"]
    S1 --> V --> S2 --> T
  end
```

DiffPin’s job is to turn the second shape into the first.

---

## Where it sits in the story

```mermaid
flowchart TB
  Agent["Agent / harness<br/>sends messages + tools"]
  Render["Chat template<br/>→ token stream"]
  DiffPin["DiffPin<br/>diff · float · pin"]
  Engine["Inference engine<br/>restore KV · prefill suffix · decode"]

  Agent --> Render --> DiffPin --> Engine
```

DiffPin does not invent a new chat API. It does not need a conversation-id header. It works from **tokens the engine already has**, plus a short memory of recent tool-bearing heads.

---

## What it is careful about

DiffPin only floats a middle hunk when that hunk looks like **ephemeral noise** (small). If the tools themselves changed, the “middle” is huge — DiffPin leaves the prompt alone. Wrong merges would be worse than a cold prefill.

```mermaid
flowchart TD
  D{Diff middle size?}
  D -->|"Small — a clock"| Float["Float it · pin the rest"]
  D -->|"Huge — tools changed"| Leave["Leave alone · new pin if needed"]
```

End-of-message markers (the “this turn is over” tokens) stay at the end of the head so the chat shape still reads as a coherent system turn.

The rewrite head stops at the **first end-of-message marker**, not at the PrefixCache chat boundary. Those boundaries sit *after* the next role-start; cutting there would let a floated middle cross into the user turn. With no chat boundaries (custom templates), DiffPin does not rewrite.

---

## What you should feel in production

| Moment | Without DiffPin | With DiffPin |
|---|---|---|
| First tool-heavy turn | Pay for the long head | Same — fill the pin once |
| Next turn, same tools, new clock | Pay for the long head again | Restore pin; prefill clock + new text |
| Multi-chat noise | Long head may get evicted | Protected pin prefers to stay |

The win is wall-clock and GPU time: tens of seconds of head prefill collapsing to a short suffix prefill when the stable block hits.

---

## Relation to other ideas

```mermaid
mindmap
  root((Reuse agent KV))
    DiffPin
      Diff finds volatility
      Float to make a contiguous pin
      No special client headers
    Harness layout
      Put clocks in a later message
      Ideal when clients cooperate
    Slot / protect policy
      Keep the pin from being thrashing away
      Complements DiffPin
```

DiffPin is the engine-side invention when clients bury clocks inside one big system blob. A disciplined harness still helps; DiffPin is the safety net that makes the common messy case pin-friendly anyway.

---

## Naming

| Term | Meaning |
|---|---|
| **DiffPin** | The invention: diff → float volatility → pin contiguous stable prefix |
| **Pin** | The KV snapshot of that stable prefix |
| **Float** | Moving the volatile hunk after the stable block |
| **Pin-friendly** | Stable tokens form one uninterrupted prefix |

---

## Closing picture

```mermaid
flowchart TB
  Q["Question: why did warm cache miss<br/>when only the date changed?"]
  A["Answer: the date sat inside the pin key"]
  I["Invention: DiffPin"]
  M["Method: diff the head, float the date,<br/>pin the paper that stayed the same"]

  Q --> A --> I --> M
```

Agents will keep sending large, almost-stable prompts. DiffPin makes “almost” good enough for KV reuse — by shaping the prompt so the cache can see the stability that was there all along.
