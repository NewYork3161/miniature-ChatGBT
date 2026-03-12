# Mini ChatGBT

![Mini ChatGBT](Images/ChatGBT.png)

---

# Overview

Mini ChatGBT is a modular open-source conversational AI framework designed to demonstrate the architecture and engineering principles behind modern language models.

The project implements a full conversational AI pipeline including:

• dataset preparation  
• tokenization  
• vocabulary encoding  
• neural language model training  
• inference generation  
• conversation memory  
• vector similarity retrieval  
• external information lookup  

The system is designed for transparency and experimentation. Each component of the AI system is separated into independent modules so contributors can study and modify specific parts of the architecture.

This repository is intended to serve as a **community-driven conversational AI system** where developers can collaboratively improve the model architecture, training process, and dataset quality.

---

# Core Design Philosophy

Modern AI systems are extremely large and difficult to study. Mini ChatGBT intentionally implements a simplified architecture that exposes every stage of the pipeline.

The system follows several design principles:

**Modularity**

Each major subsystem is implemented in a separate module.

**Transparency**

All model behavior can be traced and inspected.

**Reproducibility**

Training and inference pipelines are deterministic and reproducible.

**Extensibility**

Developers can easily swap components such as tokenizers, training loops, or models.

---

# Full System Pipeline

The conversational AI pipeline follows a multi-stage architecture.
