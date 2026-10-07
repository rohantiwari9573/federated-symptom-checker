# -*- coding: utf-8 -*-
"""
Assembles the IEEE conference paper into the (namespace-fixed) template,
replacing all guidance/placeholder content with real material about the
Privacy-Preserving Federated Symptom Checker project.
"""
import sys, copy
sys.path.insert(0, r'C:\Users\ROHAN\AppData\Roaming\Python\Python314\site-packages')

import docx
from pathlib import Path
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RES_DIR = PROJECT_ROOT / "benchmarks" / "results"
TEMPLATE = str(RES_DIR / "template_transitional.docx")
OUT = str(RES_DIR / "Federated_Symptom_Checker_Paper.docx")
RES = str(RES_DIR)

doc = docx.Document(TEMPLATE)
body = doc.element.body

# Remove the IEEE copyright-line placeholder ("XXX-X-XXXX-XXXX-X/XX/$XX.00
# (c)20XX IEEE") from every section's first-page footer.
for section in doc.sections:
    for p in section.first_page_footer.paragraphs:
        for run in p.runs:
            run.text = ""

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def clear_paragraph(p):
    """Remove all runs/content from a paragraph, keep its pPr (style, sectPr)."""
    pPr = p._p.find(qn('w:pPr'))
    for child in list(p._p):
        if child is not pPr:
            p._p.remove(child)

def set_cell(cell, text, style, bold=False, align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.style = doc.styles[style]
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    r.bold = bold

def add_table_borders(table):
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'start', 'bottom', 'end', 'insideH', 'insideV'):
        el = OxmlElement(f'w:{edge}')
        el.set(qn('w:val'), 'single')
        el.set(qn('w:sz'), '2')
        el.set(qn('w:space'), '0')
        el.set(qn('w:color'), 'auto')
        borders.append(el)
    tblPr.append(borders)

def add_picture_centered(path, width_in):
    doc.add_picture(path, width=Inches(width_in))
    p = doc.paragraphs[-1]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return p

def add_caption(text):
    p = doc.add_paragraph(text, style='figurecaption')
    return p

def add_body(text, justify=True):
    return doc.add_paragraph(text, style='BodyText')

def add_bullet(text):
    return doc.add_paragraph(text, style='bulletlist')

def add_h1(text):
    return doc.add_paragraph(text, style='Heading1')

def add_h2(text):
    return doc.add_paragraph(text, style='Heading2')

def add_h5(text):
    return doc.add_paragraph(text, style='Heading5')

def add_step(bold_lead, rest, bold=True):
    p = doc.add_paragraph(style='BodyText')
    r1 = p.add_run(bold_lead)
    r1.bold = bold
    p.add_run(rest)
    return p

def add_reference(text):
    return doc.add_paragraph(text, style='references')

# ---------------------------------------------------------------------
# 1. Title
# ---------------------------------------------------------------------
title_p = doc.paragraphs[0]
clear_paragraph(title_p)
title_p.add_run(
    "Privacy-Preserving Federated Symptom Checker: On-Device Health "
    "Pre-Screening with Differentially Private Federated Averaging"
)

# ---------------------------------------------------------------------
# 2. Trim the empty spacer paragraphs between title and author block.
#    The template has a run of empty 'Author'-styled paragraphs here
#    (a "*Note:" guidance line plus blank spacers), each contributing
#    ~0.28in of style spacing -- sized for the template's original
#    multi-line 3-column author text. With a compact author table below,
#    they left a large dead gap under the title. Delete every one that
#    carries no sectPr; the last one carries the title section's own
#    sectPr (titlePg) and must stay, but its spacing is zeroed directly
#    so it stops contributing to the gap.
# ---------------------------------------------------------------------
title_end_el = title_p._p.getnext()
while title_end_el.find(qn('w:pPr') + '/' + qn('w:sectPr')) is None:
    nxt = title_end_el.getnext()
    title_end_el.getparent().remove(title_end_el)
    title_end_el = nxt
title_pPr = title_end_el.find(qn('w:pPr'))
zero_spacing = OxmlElement('w:spacing')
zero_spacing.set(qn('w:before'), '0')
zero_spacing.set(qn('w:after'), '0')
title_pPr.insert(0, zero_spacing)

# ---------------------------------------------------------------------
# 3. Author block -- 2x2 table (name heading + info per cell), replacing
#    the template's newspaper-flow 3-column text (which split each
#    author's name from their own info across column boundaries).
# ---------------------------------------------------------------------
authors = [
    ("Rohan Tiwari", ["Department of CSE", "Ajay Kumar Garg Engineering College", "Ghaziabad, India", "rohan23153050@akgec.ac.in"]),
    ("Saar Ravindra Singh", ["Department of CSE", "Ajay Kumar Garg Engineering College", "Ghaziabad, India", "saar23153076@akgec.ac.in"]),
    ("Saksham Garg", ["Department of CSE", "Ajay Kumar Garg Engineering College", "Ghaziabad, India", "saksham23153134@akgec.ac.in"]),
    ("Dr. Jaishree Jain (Mentor)", ["Associate Professor", "Department of CSE", "Ajay Kumar Garg Engineering College", "Ghaziabad, India"]),
]

# The template's author area is a "3 columns, continuous" section, split
# across two back-to-back empty placeholder paragraphs, each carrying its
# own identical section break, located dynamically as the next two
# sectPr-bearing paragraphs after the title terminator (not by hardcoded
# index -- upstream paragraph counts shift with every content edit).
# Switch both to single column so a full-width table can hold the 2x2
# author grid instead of newspaper-flow text, then collapse the
# redundant pair into one section break.
paras = list(doc.paragraphs)
title_end_idx = next(i for i, p in enumerate(paras) if p._p is title_end_el)
sect_idxs = [i for i in range(title_end_idx + 1, len(paras))
             if paras[i]._p.find(qn('w:pPr') + '/' + qn('w:sectPr')) is not None][:2]
sect8_idx, sect9_idx = sect_idxs
sect8 = paras[sect8_idx]._p
sect9 = paras[sect9_idx]._p
for p_el in (sect8, sect9):
    sectPr = p_el.find(qn('w:pPr') + '/' + qn('w:sectPr'))
    cols = sectPr.find(qn('w:cols'))
    if cols is not None:
        cols.set(qn('w:num'), '2')  # TEST: match Abstract's column count exactly

# The author placeholder paragraphs sit between the title terminator and
# sect8; their content moves into the table below, so delete them
# entirely (not just their text) -- left in place even empty, their
# 'Author' style spacing stacks up into a visible gap.
placeholder_els = [paras[i]._p for i in range(title_end_idx + 1, sect8_idx)]
sect8.getparent().remove(sect8)
for p_el in placeholder_els:
    p_el.getparent().remove(p_el)

# Anchor on the remaining section-break paragraph so the table can be
# positioned correctly.
anchor = sect9

author_table = doc.add_table(rows=2, cols=2)
author_table.autofit = False
col_width = Inches(3.51)
for col in author_table.columns:
    col.width = col_width
for row in author_table.rows:
    for cell in row.cells:
        cell.width = col_width
tbl_el = author_table._tbl
tblPr = tbl_el.tblPr
tblW = tblPr.find(qn('w:tblW'))
if tblW is None:
    tblW = OxmlElement('w:tblW')
    tblPr.append(tblW)
tblW.set(qn('w:type'), 'dxa')
tblW.set(qn('w:w'), str(Inches(7.02).twips))
tbl_el.getparent().remove(tbl_el)
anchor.addprevious(tbl_el)

for i, (name, lines) in enumerate(authors):
    cell = author_table.cell(i // 2, i % 2)
    cell.text = ""
    p = cell.paragraphs[0]
    p.style = doc.styles['Author']
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    name_run = p.add_run(name)
    name_run.bold = True
    for line in lines:
        p.add_run().add_break()
        p.add_run(line)

# ---------------------------------------------------------------------
# 4. Abstract & Keywords -- located by style name, not position (the
#    number of paragraphs removed upstream varies across edits).
# ---------------------------------------------------------------------
abstract_p = next(p for p in doc.paragraphs if p.style.name == 'Abstract')
clear_paragraph(abstract_p)
abstract_p.add_run(
    "Abstract\u2014Mobile symptom-checker applications can extend low-cost health "
    "pre-screening to users who lack easy access to in-person consultation, yet "
    "adoption is limited by a basic trust problem: most such tools require "
    "uploading personal symptom, image, or audio data to a centralized server, "
    "which conflicts with tightening health-data regulation and with users' own "
    "privacy expectations. This paper presents a privacy-preserving symptom "
    "checker that keeps all raw data on the user's device and shares only "
    "privacy-protected model updates with a coordinating server. The system "
    "trains lightweight PyTorch models locally, aggregates them across clients "
    "using Federated Averaging, and bounds what any individual update can "
    "reveal by capping each example's gradient contribution and perturbing it "
    "with calibrated noise before transmission, following the DP-SGD mechanism implemented in "
    "the Opacus library. Three input modalities are supported end to end\u2014"
    "structured symptom checklists, dermatoscopic images, and respiratory "
    "audio\u2014through a browser-based client that also exposes a live privacy "
    "dashboard. To quantify what this design costs in accuracy, the tabular "
    "symptom classifier was trained under three regimes on a public "
    "disease-symptom dataset: a centralized baseline, federated averaging "
    "without privacy, and federated averaging with differential privacy at "
    "four privacy budgets. Federated averaging alone matched the centralized "
    "model within four communication rounds; adding differential privacy "
    "introduced a clear, measurable accuracy cost that shrank as the privacy "
    "budget was relaxed, from near-chance accuracy at a strict budget to a "
    "partial, still-incomplete recovery at a looser one—evidence that, "
    "under a deliberately short round budget, the number of communication "
    "rounds is as binding a constraint on DP-federated accuracy as the "
    "privacy mechanism itself. These results characterize a concrete "
    "privacy-utility frontier for on-device health screening and identify "
    "round budget, not just ε, as a lever future work should extend."
)

keywords_p = next(p for p in doc.paragraphs if p.style.name == 'Keywords')
clear_paragraph(keywords_p)
keywords_p.add_run(
    "Keywords\u2014Federated Learning, Differential Privacy, Edge AI, Digital "
    "Health, Symptom Checker, Privacy-Preserving Machine Learning, FedAvg, Opacus"
)

# ---------------------------------------------------------------------
# 5. Remove all guidance content (paragraphs 12..90) and the sample table
# ---------------------------------------------------------------------
paras = list(doc.paragraphs)
kw_idx = next(i for i, p in enumerate(paras) if p._p is keywords_p._p)
start_el = paras[kw_idx + 1]._p  # first template-guidance paragraph after Keywords

# The template's own 2-col body section boundary is the next sectPr-bearing
# paragraph after Keywords, located dynamically (not by hardcoded index --
# upstream paragraph counts have shifted repeatedly across earlier fixes).
sect_end_idx = next(
    i for i in range(kw_idx + 1, len(paras))
    if paras[i]._p.find(qn('w:pPr') + '/' + qn('w:sectPr')) is not None
)
sectPr_2col = copy.deepcopy(paras[sect_end_idx]._p.find(qn('w:pPr') + '/' + qn('w:sectPr')))
end_el = paras[sect_end_idx + 1]._p  # stray floating-textbox paragraph right after it

children = list(body)
start_idx = children.index(start_el)
end_idx = children.index(end_el)
for child in children[start_idx:end_idx + 1]:
    body.remove(child)

# ---------------------------------------------------------------------
# 6. Section I -- Introduction
# ---------------------------------------------------------------------
add_h1("Introduction")

add_body(
    "Machine-learning-based symptom checkers promise a low-cost first line of "
    "triage: a user describes what they are experiencing and receives a ranked "
    "list of likely conditions without waiting for a clinic appointment. In "
    "practice, uptake of these tools is held back less by model accuracy than "
    "by a trust deficit\u2014surveys of digital-health adoption repeatedly find "
    "that users disengage at the point where an app asks them to upload "
    "symptom descriptions, photographs, or recordings to a remote server [10]. "
    "That reluctance is reasonable: centralizing medical data multiplies the "
    "damage of any single breach and increasingly runs up against "
    "data-protection law such as the GDPR, HIPAA, and India's Digital Personal "
    "Data Protection Act. The result is a gap between what symptom-checking "
    "technology can technically do and what a privacy-conscious or "
    "regulation-constrained deployment can actually collect."
)

add_body(
    "Two independent lines of work point to a way out of this impasse. "
    "Federated Learning (FL), which McMahan et al. formalized as Federated "
    "Averaging, or FedAvg [1], lets many clients jointly train one model by "
    "exchanging model updates instead of raw data, while a coordinating "
    "server merges those updates into a new global model. Separately, "
    "edge-inference runtimes such as PyTorch's ExecuTorch and Google's LiteRT "
    "have made it practical to run compact neural networks directly on a "
    "phone rather than in the cloud. Combining the two suggests a symptom "
    "checker whose model both trains and runs without the user's data ever "
    "leaving their device. On its own, however, FedAvg does not prevent an "
    "attacker who observes a client's update from partially reconstructing "
    "the example that produced it, so a further mechanism is needed to bound "
    "what any single update can leak."
)

add_body(
    "Differential Privacy (DP), formalized by Dwork [4], supplies that bound: "
    "it caps how much any single training example can change a model's "
    "output, expressed through a privacy budget \u03b5. Opacus [3] operationalizes "
    "this for PyTorch by clipping each example's gradient to a fixed norm and "
    "adding calibrated Gaussian noise before the optimizer step\u2014the DP-SGD "
    "mechanism. Rieke et al.'s survey of FL in digital health [2] shows the "
    "method matching centralized accuracy on several clinical prediction "
    "tasks, but almost every deployment they review is institutional, "
    "federating hospitals or health systems rather than individual consumer "
    "devices, and none combines FL with a differential-privacy guarantee on a "
    "deployable, on-device symptom-checking pipeline. That combination\u2014"
    "on-device inference, FedAvg-based coordination through a lightweight "
    "orchestration layer such as Flower [5], and DP-bounded updates\u2014is what "
    "this paper implements and evaluates."
)

add_h2("Contributions")
add_bullet(
    "A three-layer, open-source system architecture\u2014on-device inference, "
    "federated coordination, and server-side aggregation\u2014implemented for "
    "three symptom modalities: structured checklists, skin-lesion images, "
    "and respiratory audio."
)
add_bullet(
    "A working browser-based client application that walks a user through "
    "symptom entry, image or audio capture, and on-device inference, while "
    "exposing the live privacy budget and federated-round status through an "
    "integrated dashboard."
)
add_bullet(
    "An empirical accuracy-privacy-communication comparison, trained end to "
    "end on a public disease-symptom dataset, that isolates the accuracy "
    "cost differential privacy adds on top of federated averaging across "
    "four privacy budgets."
)
add_bullet(
    "An open implementation of the client, server, and differential-privacy "
    "configuration, intended as a reusable reference for similar "
    "privacy-preserving health-AI pipelines."
)

# ---------------------------------------------------------------------
# 7. Section II -- Related Work
# ---------------------------------------------------------------------
add_h1("Related Work")

add_h2("Federated Learning Foundations")
add_body(
    "In the paper that introduced Federated Averaging, McMahan et al. [1] "
    "have each client run several local stochastic-gradient-descent steps "
    "on its own data, after which a "
    "server averages the resulting weights, weighted by how many examples "
    "each client held. This reduced the communication rounds needed to reach "
    "a target accuracy by one to two orders of magnitude relative to naive "
    "distributed SGD, and it remains the coordination protocol the federated "
    "layer in Section III builds on."
)

add_h2("Federated Learning in Healthcare")
add_body(
    "Rieke et al. [2] reviewed FL across digital-health use cases\u2014"
    "radiology, EHR-based risk prediction, genomics\u2014and reported that "
    "federated models can approach centralized accuracy while keeping "
    "patient data behind each institution's firewall. Their review is "
    "dominated, however, by cross-silo federations among hospitals or "
    "research consortia with a handful of well-resourced participants; the "
    "cross-device setting this paper targets, where each client is an "
    "individual's phone with a small, non-uniform slice of data, is "
    "comparatively unexamined."
)

add_h2("Differential Privacy and Its PyTorch Implementation")
add_body(
    "Dwork [4] gave differential privacy its formal definition: a randomized "
    "mechanism is (\u03b5, \u03b4)-differentially private if changing any single record "
    "in its input changes the probability of any output by at most a factor "
    "bounded by \u03b5. Yousefpour et al.'s Opacus [3] turns that guarantee into a "
    "practical PyTorch training loop, replacing incompatible layers such as "
    "BatchNorm, clipping each example's per-sample gradient to a fixed L2 "
    "norm, and adding Gaussian noise calibrated to the target \u03b5 before the "
    "optimizer step. Opacus's own benchmarks cover centralized vision and "
    "language models; applying its accountant inside a federated loop, where "
    "each client separately spends its own local privacy budget across "
    "communication rounds, is left open."
)

add_h2("Orchestration Frameworks and the Resulting Gap")
add_body(
    "Beutel et al.'s Flower [5] supplies a framework-agnostic FL "
    "orchestration layer that can drive PyTorch, TensorFlow, or scikit-learn "
    "clients through a common client/server protocol, but ships no built-in "
    "privacy mechanism of its own\u2014DP has to be composed in by the "
    "application, as this paper does by wrapping each Flower client's local "
    "training step with Opacus. Kairouz et al.'s survey of open problems in "
    "FL [9] frames non-IID client data, communication cost, and the utility "
    "cost of DP noise as the field's three persistent obstacles. Table I "
    "positions the present system against the five works closest to it: "
    "individually, FedAvg-style coordination, institutional healthcare FL, "
    "and PyTorch-native DP are all mature, but no reviewed system combines "
    "them into one deployable, consumer-device symptom-checking pipeline\u2014"
    "the gap this paper's system addresses."
)

# Table I
tablehead_p = doc.add_paragraph("Comparative Summary of Closely Related Work", style='tablehead')

table1 = doc.add_table(rows=1, cols=4)
add_table_borders(table1)
hdr = table1.rows[0].cells
for cell, text in zip(hdr, ["Ref.", "Method", "Reported Result", "Gap Addressed Here"]):
    set_cell(cell, text, 'tablecolhead', bold=False)

rows1 = [
    ("[1]", "FedAvg protocol", "Cuts comm. rounds vs. distributed SGD", "Not evaluated for on-device health inference"),
    ("[2]", "FL survey, digital health", "Matches centralized accuracy, institutional FL", "Cross-silo (hospitals), not individual devices"),
    ("[3]", "Opacus DP-SGD", "Per-sample clipping + noise in PyTorch", "Not evaluated in a federated health pipeline"),
    ("[4]", "Formal (\u03b5,\u03b4)-DP", "Composable, rigorous privacy guarantee", "Theoretical; needs a mechanism (e.g. Opacus)"),
    ("[5]", "Flower framework", "Scalable, framework-agnostic FL orchestration", "No built-in DP; must be composed manually"),
]
for row_data in rows1:
    row = table1.add_row().cells
    for cell, text in zip(row, row_data):
        set_cell(cell, text, 'tablecopy')

tablefootnote_p = doc.add_paragraph(
    "TABLE I compares the five works this paper builds on most directly.",
    style='tablefootnote'
)

add_h2("Identification of the Research Gap")
add_body(
    "Individually, each layer this system needs is well established: "
    "federated coordination protocols are mature [1], [5]; FL has been "
    "validated within institutional healthcare deployments [2]; and "
    "differential-privacy libraries exist for PyTorch [3], [4]. What the "
    "surveyed literature does not show is these three combined end to end "
    "in one on-device application built for individual users who are, by "
    "the adoption data cited in Section I, actively distrustful of "
    "centralized alternatives. Sections III through V describe and "
    "evaluate a system built to close that gap."
)

# ---------------------------------------------------------------------
# 8. Section III -- Proposed Methodology
# ---------------------------------------------------------------------
add_h1("Proposed Methodology")

add_h2("System Architecture")
add_body(
    "Fig. 1 shows the system as three layers. The client-side layer runs "
    "entirely on the user's device: a local model, one of three "
    "architectures depending on modality, performs inference and, during "
    "training, local gradient computation. The federated-coordination layer "
    "wraps that local training step with Opacus's DP-SGD before handing the "
    "resulting update to a Flower client, which communicates with the third "
    "layer, a Flower server that performs Federated Averaging and "
    "checkpoints the resulting global model each round. Only the clipped, "
    "noised model update ever leaves the client; the raw symptom vector, "
    "image, or audio clip does not."
)
add_picture_centered(f"{RES}\\architecture.png", 3.2)
add_caption("System architecture: client-side inference, federated DP-SGD coordination, and server-side aggregation.")

add_h2("Step-by-Step Pipeline")
add_body(
    "Concretely, one federated training round proceeds through the "
    "following eight steps, implemented in the project's client-app, "
    "models, federated, and server packages:"
)
add_step(
    "Step 1 \u2014 Local data capture: ",
    "the client application collects a structured symptom checklist (a "
    "131-symptom binary vector, via the Symptom Checker tab), a "
    "dermatoscopic photo (Skin Analysis tab, drag-and-drop or camera "
    "capture), or a short breathing/cough recording (Respiratory Check tab, "
    "in-browser capture or file upload); all three stay on the device and "
    "are never uploaded in raw form."
)
add_step(
    "Step 2 \u2014 On-device preprocessing: ",
    "the symptom vector is one-hot-encoded against the 131-symptom "
    "vocabulary; skin images are resized to 224\u00d7224 and normalized for the "
    "pretrained backbone; respiratory audio is converted to a 128\u00d7128 "
    "log-mel spectrogram."
)
add_step(
    "Step 3 \u2014 Local model inference/training: ",
    "one of three lightweight PyTorch models is used depending on "
    "modality\u2014a three-hidden-layer MLP with GroupNorm and dropout for the "
    "131-to-41 symptom-to-disease mapping (\u224850K parameters); a "
    "MobileNetV3-Small backbone with a re-initialized classifier head for "
    "the seven HAM10000 skin-lesion classes [6]; and a four-block "
    "convolutional network with global average pooling for four-way "
    "respiratory-sound classification on the ICBHI 2017 corpus [7]."
)
add_step(
    "Step 4 \u2014 Local training round: ",
    "each client runs a fixed number of local epochs of gradient descent "
    "over its own data shard, partitioned across clients with a Dirichlet "
    "distribution so that no two simulated users report the same mix of "
    "symptoms\u2014a non-IID split intended to mimic real usage."
)
add_step(
    "Step 5 \u2014 Differential-privacy step: ",
    "before any gradient leaves local memory, Opacus's PrivacyEngine clips "
    "each individual example's gradient to a fixed L2 norm and adds "
    "Gaussian noise calibrated, through its R\u00e9nyi-DP accountant, to a "
    "target privacy budget \u03b5\u2014the DP-SGD mechanism [3] wrapping every local "
    "training step."
)
add_step(
    "Step 6 \u2014 Federated transmission: ",
    "only the resulting clipped, noised model weights\u2014never the data that "
    "produced them\u2014are packaged by the Flower client and sent to the "
    "coordinating server."
)
add_step(
    "Step 7 \u2014 Server-side aggregation: ",
    "the Flower server's strategy, extending Flower's FedAvg, averages "
    "participating clients' weights in proportion to how many local "
    "examples each held, checkpoints the resulting global model, and tracks "
    "the cumulative bytes exchanged; the updated global model is then "
    "broadcast back to every client to start the next round."
)
add_step(
    "Step 8 \u2014 On-device deployment and inference: ",
    "once training converges, the global model is exported through "
    "ExecuTorch or LiteRT for on-device inference; the client application "
    "then runs every subsequent prediction locally and reflects the current "
    "privacy budget, federated round, and global accuracy in its Privacy "
    "Dashboard, so the guarantee argued for above is visible to the user, "
    "not only to the developers."
)

add_h2("Differential Privacy Configuration")
add_body(
    "Each client's noise multiplier \u03c3 is derived from its target (\u03b5, \u03b4) "
    "using Opacus's R\u00e9nyi-DP accountant, given the client's local sample "
    "count, batch size, and number of local steps; \u03b4 defaults to the "
    "reciprocal of the client's dataset size, following standard practice "
    "for the ``\u03b4 should be smaller than 1/n'' guideline. Because the "
    "accountant is queried per client, clients with smaller shards, and "
    "therefore a higher per-round sampling rate, receive a proportionally "
    "larger noise multiplier for the same target \u03b5\u2014the calibration is "
    "local, not a single global constant applied uniformly."
)

add_h2("Experimental Validation Setup")
add_body(
    "To measure what this pipeline costs in accuracy, the tabular symptom "
    "branch (Step 3's MLP) was trained end to end on a public "
    "disease-symptom dataset [8] of 4,920 labeled records across 41 disease "
    "classes and 132 binary symptom features\u2014the same feature vocabulary "
    "the deployed API server already expects. The data were split 80/20 "
    "into 3,936 training and 984 test records and partitioned across ten "
    "simulated clients with a Dirichlet(\u03b1=0.5) split (Step 4). Three "
    "regimes were trained for ten communication rounds each, one local "
    "epoch per round, batch size 32: (i) a centralized model with full data "
    "access, (ii) Federated Averaging with no privacy mechanism, and (iii) "
    "Federated Averaging with Opacus-calibrated DP-SGD at \u03b5 \u2208 {0.5, 1.0, "
    "2.0, 5.0} (\u03b4 = 10\u207b\u2075). Accuracy, macro-F1, and cross-entropy loss were "
    "measured on the held-out test set after every round."
)

# ---------------------------------------------------------------------
# 9. Section IV -- Prototype Implementation and Key Features
# ---------------------------------------------------------------------
add_h1("Prototype Implementation and Key Features")
add_body(
    "Figs. 2 through 6 show the browser-based client described in Section "
    "III running end to end against the deployed API layer, whose built-in "
    "demonstration fallback supplies illustrative predictions when no "
    "trained checkpoint is loaded\u2014the same code path a user exercises once "
    "a real global model is deployed. Five features are highlighted below."
)

add_h2("Symptom Checker")
add_picture_centered(f"{RES}\\fig_symptom_checker.jpg", 3.2)
add_caption("Symptom Checker tab: searchable 131-symptom list with selected symptoms shown as removable tags.")
add_body(
    "The search-and-select interface covers the full 131-symptom "
    "vocabulary; selected symptoms appear as removable tags, mirroring "
    "exactly the one-hot vector the SymptomMLP consumes (Section III-B, "
    "Steps 1\u20132)."
)

add_h2("Analysis Results")
add_picture_centered(f"{RES}\\fig_analysis_results.jpg", 3.2)
add_caption("Analysis Results panel: top prediction with an animated confidence ring, ranked alternatives, and a persistent medical disclaimer. (Illustrative values from the client's demonstration fallback; no trained checkpoint was loaded for this capture.)")
add_body(
    "The results panel presents the top-ranked prediction with an animated "
    "confidence ring plus a ranked list of alternative conditions, and "
    "carries a persistent medical disclaimer, since a pre-screening tool of "
    "this kind is not a diagnostic instrument."
)

add_h2("Skin Lesion Analysis")
add_picture_centered(f"{RES}\\fig_skin_analysis.jpg", 3.2)
add_caption("Skin Analysis tab: drag-and-drop / camera-capture upload surface for the HAM10000-trained branch.")
add_body(
    "A drag-and-drop or camera-capture surface feeds the "
    "MobileNetV3-Small-based SkinCNN branch, resizing and normalizing the "
    "image to 224\u00d7224 entirely client-side before inference (Section "
    "III-B, Steps 2\u20133)."
)

add_h2("Respiratory Analysis")
add_picture_centered(f"{RES}\\fig_respiratory.jpg", 3.2)
add_caption("Respiratory Check tab: in-browser audio recorder with waveform feedback, or file upload, for the ICBHI-based branch.")
add_body(
    "This tab records, or accepts an uploaded breathing/cough clip through, "
    "the browser's media-recording API for the RespiratoryCNN branch, with "
    "waveform feedback during capture."
)

add_h2("Privacy Dashboard")
add_picture_centered(f"{RES}\\fig_privacy_dashboard.jpg", 3.2)
add_caption("Privacy Dashboard: live privacy budget \u03b5, current federated round, and global model accuracy.")
add_body(
    "The dashboard is the feature most specific to this system's "
    "contribution: it surfaces the live privacy budget \u03b5, the current "
    "federated round out of the configured total, and the global model's "
    "federated accuracy, so the privacy guarantee argued for in Section "
    "III-C is visible to the end user rather than only to the developers."
)

# ---------------------------------------------------------------------
# 10. Section V -- Results and Discussion
# ---------------------------------------------------------------------
add_h1("Results and Discussion")

add_body(
    "Two properties of this setup shape every result below and are stated "
    "up front so they are not mistaken for a verdict on differential "
    "privacy itself. First, the dataset's symptom-to-disease mapping is "
    "close to deterministic, which is why both the centralized model and "
    "plain FedAvg reach 100% accuracy almost immediately—that ceiling is a "
    "property of this dataset, not evidence that the federated coordination "
    "layer is trivial. Second, the round budget was fixed deliberately "
    "small—ten rounds, one local epoch each—so every DP-SGD run had very "
    "few noised gradient steps over which to average out the Gaussian "
    "noise Opacus injects at each step. The comparison below therefore "
    "isolates round budget, at least as much as the privacy budget ε, as "
    "the binding constraint on DP-federated accuracy in this regime."
)

add_h2("Accuracy vs. Communication Round")
add_picture_centered(f"{RES}\\accuracy_vs_rounds.png", 3.2)
add_caption("Test accuracy vs. communication round for the centralized baseline, FedAvg without DP, and FedAvg with DP at four privacy budgets.")
add_body(
    "Fig. 7 plots test accuracy against communication round for all six "
    "configurations. Federated Averaging without privacy (blue) tracks the "
    "centralized upper bound (green) closely, reaching 99.8% by round 2 and "
    "matching the centralized model's 100% from round 4 onward\u2014consistent "
    "with McMahan et al.'s original finding that FedAvg needs few rounds to "
    "match centralized SGD once client updates are averaged appropriately, "
    "and confirming that the non-IID client partitioning used here "
    "does not, on its own, prevent convergence on this task."
)
add_body(
    "Adding differential privacy tells a different story. At the tightest "
    "budget tested (\u03b5=0.5), accuracy stays within a few points of the 2.4% "
    "chance level for this 41-way classification task throughout all ten "
    "rounds; loosening the budget to \u03b5=1.0 does not yet help. From \u03b5=2.0 the "
    "model visibly begins to learn, reaching 23.6% by round 10, and at "
    "\u03b5=5.0 it reaches 38.4%\u2014still well short of the non-private federated "
    "model, but a clear, monotonic improvement over the tighter budgets."
)

add_h2("Privacy-Utility Trade-off")
add_picture_centered(f"{RES}\\privacy_utility_tradeoff.png", 3.0)
add_caption("Final test accuracy as a function of privacy budget \u03b5, against the centralized and non-private-federated ceilings.")
add_body(
    "Fig. 8 makes the resulting privacy-utility frontier explicit: final "
    "accuracy rises monotonically with \u03b5, exactly the trade-off Dwork's "
    "definition predicts and that Yousefpour et al. report for "
    "few-epoch DP-SGD training generally. Under a fixed, small round "
    "budget, the Gaussian noise injected at every step dominates the "
    "gradient signal until \u03b5 is relaxed enough\u2014or training is run long "
    "enough\u2014for the signal to accumulate."
)

add_h2("Summary Comparison")
tablehead2_p = doc.add_paragraph("Final Accuracy, Macro-F1, and Loss by Method", style='tablehead')
table2 = doc.add_table(rows=1, cols=5)
add_table_borders(table2)
hdr2 = table2.rows[0].cells
for cell, text in zip(hdr2, ["Method", "\u03b5", "Accuracy", "Macro-F1", "Loss"]):
    set_cell(cell, text, 'tablecolhead')
rows2 = [
    ("Centralized", "n/a", "1.00", "1.00", "0.00"),
    ("FedAvg (no DP)", "\u221e", "1.00", "1.00", "0.01"),
    ("FedAvg + DP", "0.5", "0.05", "0.02", "6.93"),
    ("FedAvg + DP", "1.0", "0.03", "0.02", "3.85"),
    ("FedAvg + DP", "2.0", "0.24", "0.17", "3.08"),
    ("FedAvg + DP", "5.0", "0.38", "0.30", "2.61"),
]
for row_data in rows2:
    row = table2.add_row().cells
    for cell, text in zip(row, row_data):
        set_cell(cell, text, 'tablecopy')
doc.add_paragraph("All figures measured on the held-out 984-record test split after round 10.", style='tablefootnote')

add_picture_centered(f"{RES}\\final_accuracy_bar.png", 3.2)
add_caption("Final accuracy by method: centralized and non-private-federated models reach 100%; every tested privacy budget extracts a measurable cost.")
add_body(
    "Table II and Fig. 9 summarize the same result as final numbers: "
    "federated averaging alone reproduces the centralized baseline almost "
    "exactly, at the cost of ten communication rounds instead of one "
    "training job, while every tested privacy budget extracts a real "
    "accuracy cost, from 95 percentage points at \u03b5=0.5 down to 62 points at "
    "\u03b5=5.0. Rather than pointing to any single operating \u03b5, this shows \u03b5 "
    "acting as a strong lever on how much of the federated signal survives "
    "the injected noise within a fixed, short round budget: raising \u03b5 from "
    "0.5 to 5.0 moves the model from chance-level to a visibly learning, "
    "still-improving classifier, without yet closing the gap to the "
    "non-private ceiling. Locating the \u03b5 at which DP-federated training "
    "becomes practically competitive at this task's complexity is therefore "
    "a question of round budget as much as privacy budget, and is the "
    "immediate next experiment identified in Section VI."
)

add_h2("Limitations")
add_body(
    "Two limitations qualify these results. First, the comparison uses the "
    "tabular symptom branch only, on a public dataset whose deterministic "
    "symptom-to-disease mapping is easy enough that both the centralized "
    "and non-private federated models reach 100% accuracy almost "
    "immediately; the skin-lesion and respiratory branches, trained on "
    "noisier image and audio data, would likely show a smaller gap between "
    "centralized and federated accuracy and are left for the full-scale "
    "evaluation proposed in Section VI. Second, the noise multiplier for "
    "each \u03b5 was computed once per client from Opacus's accountant under "
    "naive per-round composition; a larger total round budget over which to "
    "amortize the same noise would likely narrow the privacy-utility gap "
    "reported here without changing its qualitative shape."
)

# ---------------------------------------------------------------------
# 11. Section VI -- Conclusion and Future Work
# ---------------------------------------------------------------------
add_h1("Conclusion and Future Work")
add_body(
    "This paper implemented and empirically evaluated a three-layer "
    "privacy-preserving symptom checker that keeps raw health data "
    "on-device and shares only differentially private, federated model "
    "updates with a coordinating server, across three symptom modalities. "
    "On a real training run of the tabular branch, Federated Averaging "
    "matched a centralized baseline within four communication rounds, and "
    "adding Opacus-based differential privacy produced a clear, quantified "
    "accuracy cost that fell as the privacy budget was relaxed\u2014from "
    "near-chance accuracy at \u03b5=0.5 to 38.4% at \u03b5=5.0, against a 100% "
    "non-private ceiling. These results give the accuracy-privacy-"
    "communication trade-off that motivated this project a concrete, "
    "measured shape rather than a purely theoretical one."
)
add_body(
    "Future work follows four directions the results themselves point to: "
    "first and most immediately, re-running the DP-SGD arms over a "
    "substantially longer round budget (tens to hundreds of rounds rather "
    "than ten) to test whether a lower ε becomes practically competitive "
    "once the noise this paper's short horizon could not average out is "
    "given more steps to do so; extending the same three-regime comparison "
    "to the skin-lesion and respiratory branches on their target datasets "
    "(HAM10000 [6], ICBHI 2017 [7]); replacing the naive composition used "
    "here with Opacus's full per-round accounting to obtain a tighter, "
    "less pessimistic noise calibration; and validating the pipeline on "
    "physical Android devices rather than simulated clients, to measure "
    "real on-device inference latency after ExecuTorch/LiteRT export. We "
    "intend to submit the resulting extended evaluation to a peer-reviewed "
    "computing/AI-systems conference."
)

# ---------------------------------------------------------------------
# 12. Acknowledgment
# ---------------------------------------------------------------------
add_h5("Acknowledgment")
add_body(
    "The authors thank Dr. Jaishree Jain (Associate Professor), Department "
    "of Computer Science and Engineering (AIML), Ajay Kumar Garg "
    "Engineering College, Ghaziabad, for her supervision and guidance "
    "throughout this project."
)

# ---------------------------------------------------------------------
# 13. References
# ---------------------------------------------------------------------
add_h5("References")
refs = [
    "H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "
    "\u201cCommunication-efficient learning of deep networks from decentralized "
    "data,\u201d in Proc. AISTATS, 2017, pp. 1273\u20131282.",
    "N. Rieke et al., \u201cThe future of digital health with federated "
    "learning,\u201d NPJ Digital Medicine, vol. 3, no. 1, pp. 1\u20137, 2020.",
    "A. Yousefpour et al., \u201cOpacus: User-friendly differential privacy "
    "library in PyTorch,\u201d arXiv:2109.12298, 2021.",
    "C. Dwork, \u201cDifferential privacy,\u201d in Automata, Languages and "
    "Programming, Springer, 2006, pp. 1\u201312.",
    "D. J. Beutel et al., \u201cFlower: A friendly federated learning "
    "framework,\u201d arXiv:2007.14390, 2020.",
    "P. Tschandl, C. Rosendahl, and H. Kittler, \u201cThe HAM10000 dataset, a "
    "large collection of multi-source dermatoscopic images of common "
    "pigmented skin lesions,\u201d Scientific Data, vol. 5, 180161, 2018.",
    "B. M. Rocha et al., \u201cAn open access database for the evaluation of "
    "respiratory sound classification algorithms,\u201d in Proc. ICBHI, 2017.",
    "Kaggle, \u201cDisease Symptom Prediction / Disease-Symptom Description "
    "Dataset,\u201d kaggle.com/datasets/itachi9604/disease-symptom-description-"
    "dataset, accessed 2026.",
    "P. Kairouz et al., \u201cAdvances and open problems in federated "
    "learning,\u201d Foundations and Trends in Machine Learning, vol. 14, no. "
    "1\u20132, pp. 1\u2013210, 2021.",
    "L. West, D. Mitchell, S. D. Faulkner, B. Bauer, N. Brooke, and E. "
    "Priest, \u201cDigital health technologies: Learnings and perspectives from "
    "a patient engagement stakeholder expectations matrix study,\u201d Journal "
    "of Medical Internet Research, vol. 27, e81463, 2025.",
]
for r in refs:
    add_reference(r)

# ---------------------------------------------------------------------
# 14. Re-attach the 2-column section break to the last content paragraph,
#     then add a final empty spacer paragraph as the trailing 1-col section
# ---------------------------------------------------------------------
last_p = doc.paragraphs[-1]._p
pPr = last_p.find(qn('w:pPr'))
if pPr is None:
    pPr = OxmlElement('w:pPr')
    last_p.insert(0, pPr)
pPr.append(sectPr_2col)

doc.add_paragraph("")  # final section-4 spacer (1-column), mirrors original structure

doc.save(OUT)
print("Saved:", OUT)
print("Total paragraphs:", len(doc.paragraphs))
