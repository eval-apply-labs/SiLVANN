"""QWEN3.8-FLASH-NEXT (`qwen4_exp`) — DeltaNet and Qwen Sparse Attention, four residual streams through a gated residual,
512 experts of 640 and a shared one, an n-gram embedding at its second layer — the matrices on the card, the routed experts
on the CPU (NN-141).

⛳ THE FIRST PROGRAM, IN nn's WORDS ALONE (▶ src/docs/design/pending/qwen4_exp.md ⑤): `lisp/qwen4_exp.lisp`'s sites are
nn's words over each layer's table, no model package's verbs yet. A prompt is read a CHUNK at a time, a layer at a time:
each row's mixer as a position runs it, then the chunk's router picks into its hand rows and the CPU's routed experts over
them once, expert-major (ai_qwen_3's `experts_rows`). This class makes what the program reads — the collections, the load,
the buffers, the tables, the names (`_bindings`) — and the host's part of a position or a chunk: the n-gram rows
(`ngram_qwen4.NGram`, from the pack's `ngram/`) written into `ngr` (a chunk's into `ngrR`) before it runs.

A conversation's STATE is the DeltaNet layers' fp32 state and conv window, the attention layers' keys and values and
their indexer's keys (a position's, and a block's of RATIO positions), and the n-gram layer's normed values, a position
each. Qwen Sparse Attention's indexer is ai_qwen_4's verbs': up to 2,051 positions a query chooses every block, past
that its best 512 blocks and its unfinished block's positions (▶ ai_qwen_4's contracts/objects/layer.cuh); the program's
words (`verbs` false) hold the context below 2,048.
"""
import contextlib
import os
import pickle
import threading

import numpy as np

from . import expert_tier, experts_file
from .generate import STOPS_MOST, Generates
from .model_folder import NL, Refused
from .ngram_qwen4 import NGram

import nn_compressor as NC        # noqa: E402  (python/ on the path, as model_folder puts it)
import nn_unpack_model as NU      # noqa: E402

GU, DN = NL.GATE_UP, NL.DOWN
CARD = 0                  # the card whose block runs the host's programs: another card's sections are handed it
ARCHITECTURES = ("qwen4_exp_text",)
QSA_ALL = 2048            # positions at which QSA chooses every block — the indexer's budget, its unfinished block aside
CARD_FROM = 128           # a prompt's chunk from which its experts are the card's (▶ `prompt_on`)


def _half(a):
    return np.asarray(a, dtype="<f2").tobytes()


def _f32(a):
    return np.asarray(a, dtype="<f4").tobytes()


SITES = ("attn_hyper_connection", "mlp_hyper_connection")
GATED = ("hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight", "block_inject_weight.weight")
DELTANET = ("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight", "linear_attn.in_proj_a.weight",
            "linear_attn.in_proj_b.weight", "linear_attn.conv1d.weight", "linear_attn.A_log", "linear_attn.dt_bias",
            "linear_attn.norm.weight", "linear_attn.out_proj.weight")
ATTENTION = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
             "self_attn.q_norm.weight", "self_attn.k_norm.weight",
             "self_attn.indexer.index_qk_proj.weight", "self_attn.indexer.q_layernorm.weight",
             "self_attn.indexer.k_layernorm.weight")
NGRAM = ("ple.key_proj.weight", "ple.value_proj.weight", "ple.norm_conv.weight")
MOE = ("mlp.gate.weight", "mlp.shared_expert.gate_proj.weight", "mlp.shared_expert.up_proj.weight",
       "mlp.shared_expert.down_proj.weight", "mlp.shared_expert_gate.weight")

# a layer's table: a matrix three cells (codes, scales, bits), a norm or a vector one, a buffer's name one
MATRIX, ONE = 3, 1
CELLS = [("AN", SITES[0] + ".hc_norm.weight", ONE), ("AD", SITES[0] + ".input_mix_weight_down.weight", MATRIX),
         ("AU", SITES[0] + ".input_mix_weight_up.weight", MATRIX), ("AI", SITES[0] + ".block_inject_weight.weight", MATRIX),
         ("MN", SITES[1] + ".hc_norm.weight", ONE), ("MD", SITES[1] + ".input_mix_weight_down.weight", MATRIX),
         ("MU", SITES[1] + ".input_mix_weight_up.weight", MATRIX), ("MI", SITES[1] + ".block_inject_weight.weight", MATRIX),
         ("QKV", DELTANET[0], MATRIX), ("Z", DELTANET[1], MATRIX), ("A", DELTANET[2], MATRIX), ("B", DELTANET[3], MATRIX),
         ("CONV", DELTANET[4], ONE), ("ALOG", DELTANET[5], ONE), ("DTB", DELTANET[6], ONE), ("GNORM", DELTANET[7], ONE),
         ("OUT", DELTANET[8], MATRIX),
         ("Q", ATTENTION[0], MATRIX), ("K", ATTENTION[1], MATRIX), ("V", ATTENTION[2], MATRIX), ("O", ATTENTION[3], MATRIX),
         ("QN", ATTENTION[4], ONE), ("KN", ATTENTION[5], ONE),
         ("KEY", NGRAM[0], MATRIX), ("VAL", NGRAM[1], MATRIX), ("NC", NGRAM[2], ONE), ("WENG", None, ONE), ("TAP", None, 4),
         ("GATE", MOE[0], MATRIX), ("SG", MOE[1], MATRIX), ("SU", MOE[2], MATRIX), ("SD", MOE[3], MATRIX),
         ("SEG", MOE[4], MATRIX),
         ("IX", ATTENTION[6], MATRIX), ("IQN", ATTENTION[7], ONE), ("IKN", ATTENTION[8], ONE)]
AT = {}
_n = 0
for _k, _name, _w in CELLS:
    AT[_k] = _n
    _n += _w
TABLE_CELLS = _n


class Qwen4Exp(Generates):
    PROGRAM = ("qwen4_exp", "qwen4_exp_cpu", "generate")   # the pack's lisp/ files (▶ `ModelFolder.program`)
    BINDINGS = 1024
    CARDS_ARE_SEATS = True                     # several cards a conversation each, the boot's multi_user (▶ with_cards)
    # what its process holds beside its experts in RAM, for the autotune's RAM rule (▶ `experts_file.in_ram`): `MEASURED`
    #   node03 2026-11-09 (one MI50, the experts in RAM, the boot's peak of anonymous memory less the experts' 56.53 GiB),
    #   3.62 GiB at 4,096 positions and 3.57 at 196,608 — its CPU worker's pool, the embedding in RAM, the load's own —
    #   against `OWN`'s 2; 4 GiB, a CUDA card's host memory beside it (the precision's 62 GB boot was OOM-killed at 2)
    HOST_OWN = 4 << 30

    @staticmethod
    def longest_context(config):
        """The longest conversation this program holds (▶ autotune's `model_length`): the model's own — QSA's indexer is
        ai_qwen_4's verbs' (`verbs`, the default)."""
        c = config.get("text_config", config)
        return int(c.get("max_position_embeddings", QSA_ALL))

    def __init__(self, folder, machine, max_context=2048, experts_from="auto", experts_dir=None, experts_ram_gb=None,
                 experts_ahead="auto", chunk=512, experts_int8=False, sockets=1, verbs=True, rows_from=8, seats=1,
                 cards=None, prompt_on="auto", card_from=CARD_FROM, card_experts_gb=0, card_landing=1, layers=None):
        if folder.architecture not in ARCHITECTURES:
            raise Refused("%s is a %r model, not one of %s" % (folder.path, folder.architecture, ARCHITECTURES))
        if max_context > QSA_ALL and not verbs:
            raise Refused("max_context %d: past %d positions Qwen Sparse Attention chooses which blocks to attend, and its "
                          "indexer is ai_qwen_4's verbs' — verbs true" % (max_context, QSA_ALL))
        c, b = folder.config, folder.bundle
        self.folder, self.m, self.b, self.max_context = folder, machine, b, max_context
        self.H, self.LAYERS, self.HC = c["hidden_size"], c["num_hidden_layers"], c["hc_count"]
        self.S = self.HC * self.H
        self.KH, self.VH, self.LD = c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"]
        self.KEY, self.VAL = self.KH * self.LD, self.VH * self.LD
        self.CONV_CH = 2 * self.KEY + self.VAL
        self.QH, self.KVH, self.AD = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        rope = c.get("rope_parameters") or {}
        self.ROT, self.THETA = int(self.AD * rope.get("partial_rotary_factor", 1.0)), int(rope["rope_theta"])
        # QSA's indexer: its query heads and their width, a block's positions, and how many positions it chooses (▶ the
        #   attention verbs: past BUDGET / RATIO finished blocks, the best BUDGET / RATIO of them and the unfinished one's)
        self.IH, self.ID = c["indexer_n_heads"], c["indexer_head_dim"]
        self.RATIO, self.BUDGET = c["indexer_compress_ratio"], c["indexer_budget"]
        self.PICK = (self.BUDGET // self.RATIO) * self.RATIO + self.RATIO - 1
        self.ATTENDED = min(self.max_context, self.PICK)          # the most positions a query attends
        self.EXPERTS, self.TOPK, self.INTER = c["num_experts"], c["num_experts_per_tok"], c["moe_intermediate_size"]
        if c.get("shared_expert_intermediate_size", self.INTER) != self.INTER:
            raise Refused("the shared expert is another width than the routed ones")
        self.EPS = float(c["rms_norm_eps"])
        self.kinds = {l: ("f" if c["layer_types"][l] != "linear_attention" else "d") for l in range(self.LAYERS)}
        self.PLE = [i - 1 for i in c.get("ple_layer_ids") or []]
        self.CONV_K, self.DIL = c["ple_conv_kernel_size"], c["ngram_size"]
        # `layers`: the layers this boot runs, from the first, the head after the last — a check on a part of the model
        #   (▶ test/src_qwen4_exp_qsa.py); every layer by default
        self.run_layers = list(range(self.LAYERS)) if layers is None else sorted(set(int(l) for l in layers))
        if not self.run_layers or self.run_layers[0] != 0 or self.run_layers[-1] >= self.LAYERS:
            raise Refused("layers %r: from layer 0, below %d" % (layers, self.LAYERS))
        self.PLE = [l for l in self.PLE if l in self.run_layers]
        rec = b.record(0, SITES[0] + ".input_mix_weight_up.weight")
        self.R = rec.cols
        if NC.blocks_for(self.R) != ((256, 128), 0):
            raise Refused("the gated residual's rank %d is not the 256-and-128 the program rotates (pack it with "
                          "nn_pack_model's padded rank)" % self.R)
        g = pickle.load(open(os.path.join(folder.path, "global.pkl"), "rb"))
        self.global_ = g
        self.VOCAB = int(g["lm_head.weight"]["rows"])
        self.ng = NGram(folder.path)
        # a prompt's chunk: its rows through a layer before the next, the routed experts once over them (▶ `prefill`); a
        #   hand row is ai_qwen_3's — the hand, then its picks as words (AI_QWEN_3__HAND__ROW)
        #   ⚖ 2026-10-31 *"we need to set a variable with the threshold where single token is better and then for the rest
        #   do the gemv, up to the chunk we specify"*: a prompt shorter than `rows_from` positions is read a position at a
        #   time, a longer one as rows, `chunk` at a time — up to 1,024 (ai_qwen_3's experts_rows takes 16,384), the CPU
        #   reading each expert a chunk picks once: a chunk of 64 picks ~367 of a layer's 512, one of 512 nearly all.
        #   `MEASURED` node03 2026-10-31 with the rows verbs (measurements/2026-10-31_nn141_flash_next_node03.md): rows are
        #   faster from 8 positions (6 a tie), and the server's 474-position first turn takes 17.1 s at a chunk of 64, 14.1
        #   at 128, 11.7 at 256, 11.4 at 512 — the defaults, 8 and 512
        if not isinstance(chunk, int) or isinstance(chunk, bool) or not 1 <= chunk <= 1024:
            raise Refused("chunk is a number of positions from 1 to 1,024, not %r" % (chunk,))
        if not isinstance(rows_from, int) or isinstance(rows_from, bool) or rows_from < 1:
            raise Refused("rows_from is the prompt's length from which it is read as rows, 1 or more, not %r" % (rows_from,))
        self.CHUNK = min(chunk, max_context)
        self.ROWS_FROM = rows_from
        # a position's routed experts on the CPU in integers — nn's int8 gemvs, the activation quantised in blocks of 32
        #   (ai_qwen_3's AI_QWEN_3__EXP__INT8) — as a prompt's rows there always are; else exact
        self.EXPERTS_INT8 = bool(experts_int8)
        self.HAND_ROW = ((2 * (self.H + 16 + self.INTER) + 7) & ~7) + 8 * self.TOPK
        self.EOS2 = np.array([self.ng.eos, self.ng.eos], dtype="<i4").tobytes()      # `prev2` before position 0
        # ⭐ SEATS (NN-147, ⚖ 2026-10-31 seats.md ④): the conversations held on the card at once, each its own state, its
        #   streams and its MoE carry, the weights one — the card on one seat's dense part while the CPU computes another's
        #   experts, as their chains interleave on the card's queue. A seat's programs run in its own view (▶ `load`), the
        #   program's names bound to its buffers; seat 0's are the names themselves.
        if not isinstance(seats, int) or isinstance(seats, bool) or not 1 <= seats <= self.SEATS_MOST:
            raise Refused("seats is a count from 1 to %d, not %r" % (self.SEATS_MOST, seats))
        # ⭐ CARDS (`cards`, the boot's num_cards with multi_user "yes"): each card the whole dense model and its own seats, the CPU worker the
        #   routed experts of every card's — a card a worker, the CPU the worker after them. Card k's program is card 0's:
        #   its buffers, tables and values are bound under the program's own names and frozen in the view its sections run
        #   in (`names_s<seat>`), each card's bound over the one after's and card 0's last (▶ `load`). Seats are numbered across
        #   the cards, a card's `seats` after the one before's: card c's seat s is seat c·seats + s.
        cards = [0] if cards is None else list(cards)
        if not cards or len(set(cards)) != len(cards) \
                or not all(isinstance(c, int) and not isinstance(c, bool) and c >= 0 for c in cards):
            raise Refused("cards is a list of one or more cards, each once, not %r" % (cards,))
        self.CARD_IDS, self.CARDS = cards, len(cards)
        self.CPU = self.CARDS
        if self.CARDS > 1 and seats > 1:
            raise Refused("several cards hold a seat each for now: seats 1")
        # ⭐ EVERY SEAT ITS OWN CHAIN (⚖ 2026-11-04 *"simply a self-queuing exercise"*, ▶ qwen4_exp.lisp's chain): a seat's
        #   position runs as steps on its card's block, each layer's experts handed the CPU between two, so the card runs
        #   another seat's step while the CPU computes, and the CPU serves whichever seat's experts came first. Each seat
        #   steps from a thread of its own (▶ the session's `ALONE`).
        self.CARD_SEATS = seats                    # a card's seats — the program's SEATS
        # ⛳ A PROMPT IS ITS CARD'S ALONE: a chunk's n-gram rows are the card's buffer, written by the host before the chunk
        #   runs, so a card's prompts — and its `generate`s, whose answer buffers are the card's — take its lock; a seat's
        #   steps between them need none
        self._card_locks = [threading.Lock() for _ in range(self.CARDS)]
        self.SEATS, self.SEAT = self.CARDS * seats, 0
        # where the routed experts are held: RAM, the disk or the file mapped (▶ experts_file.choose)
        self.EPLAN = NL.expert_plan(b, 0, self.EXPERTS)
        self.EXPERTS_FROM = experts_file.choose(experts_from, self.EXPERTS * self.LAYERS * self.EPLAN.slot_bytes)
        # ⛔ SEVERAL CONVERSATIONS AT ONCE TAKE EVERY EXPERT IN RAM (▶ experts_file's SEATED_FROM_DISK)
        if self.SEATS > 1 and self.EXPERTS_FROM == "disk":
            raise Refused(experts_file.SEATED_FROM_DISK)
        # ⭐ A PROMPT'S EXPERTS ON ITS OWN CARD (`prompt_on`, ▶ seats.md ⑥): a chunk's routed experts computed on the card
        #   of the seat reading it — the picks its tier holds from the tier, every other streamed from the CPU's memory
        #   through a ring of the card's own a batch ahead of the GEMM (nn's tier and its stream, NN-115 ②) — the CPU left
        #   to the other cards' answers. The promotion threshold is past any chunk's picks, so a prompt promotes nothing
        #   into the tier: the ring is its staging, and the tier the positions' to fill; without `card_experts_gb` the
        #   tier is a slot a layer. "card": a chunk of `card_from` positions or more read so, a shorter one by the CPU's
        #   `experts_rows`, which copies nothing to the card; "cpu": every chunk by the CPU's; "auto" (the default):
        #   "card" where the boot allows it — a seat a card, one CPU worker, the verbs, the experts in RAM or read from the
        #   disk (a chunk's experts the disk's cache misses read from the pack into the ring, NN-165) — else "cpu".
        #   `MEASURED` node03 2026-11-02, Flash-Next, a 10 GB tier
        #   (measurements/2026-11-02_prompt_card_crossover_node03.md): the card reads a chunk faster than the CPU from 64
        #   positions on the conversation's topic once an answer has warmed the tier, from ~80 on other prose and ~180 on
        #   code; with the tier cold, from ~150 on prose and ~240 on code. From 128 (CARD_FROM) the card is ahead or within
        #   6 % in all of these but a cold tier's code (1.23×); at 8-32 positions it is behind in every one, 1.1-1.7×.
        #   From the disk (⚖ 2026-11-09, the precision's finding): `MEASURED` the RTX 5060 Ti, a 2,035-position prompt —
        #   at 32 GB 86.8 · 87.4 positions/s on the card against 29.6 on the CPU (2.9×), the token 5% faster after it; at
        #   62 GB 96.5 · 97.1 against 29.9 · 30.1 (3.2×), the token the same over 128
        #   (measurements/2026-11-08_owed_rows_precision.md ③, ②c); node03's MI50 1.4×, its token 3-5% slower after
        #   (measurements/2026-11-08_disk_stream_node03.md).
        if prompt_on not in ("auto", "cpu", "card"):
            raise Refused("prompt_on is \"auto\", \"cpu\" or \"card\", not %r" % (prompt_on,))
        if not isinstance(card_from, int) or isinstance(card_from, bool) or card_from < 1:
            raise Refused("card_from is the chunk's length from which its experts are the card's, 1 or more, not %r"
                          % (card_from,))
        allowed = seats == 1 and sockets == 1 and verbs is True and self.EXPERTS_FROM in ("ram", "disk")
        self.PROMPT_CARD = prompt_on == "card" or prompt_on == "auto" and allowed
        if self.PROMPT_CARD and (seats > 1 or sockets > 1 or not verbs):
            raise Refused("a prompt's experts on the card take a seat a card, one CPU worker and the verbs")
        self.CARD_FROM = card_from
        # ⭐ A CARD'S TIER OF ROUTED EXPERTS FOR ONE SEAT (`card_experts_gb`, NN-48's, as Qwen 3.5's): the card computes the
        #   picks it holds and the CPU the rest, the two sums added; an expert the CPU computed earns a slot by nn's
        #   qualifier and is copied whole from the CPU's memory while the card computes, `card_landing` a visit at most.
        #   ⚖ 2026-11-01 *"go ahead with the card tier knowing it will be an optimization reserved for the single seat user"*
        #   (seats.md ④: one seat a card keeps the L1 tier, more drop it). ⛳ A routed sum split between the card and the CPU
        #   repeats from boot, not from one request to the next, as the tier fills (NN-70).
        if not isinstance(card_experts_gb, (int, float)) or isinstance(card_experts_gb, bool) or card_experts_gb < 0:
            raise Refused("card_experts_gb is the card's room for a tier of experts, in GB, not %r" % (card_experts_gb,))
        if not isinstance(card_landing, int) or isinstance(card_landing, bool) or card_landing < 1:
            raise Refused("card_landing is the most promotions a visit starts, 1 or more, not %r" % (card_landing,))
        self.CARD_GB, self.CARD_LANDING = float(card_experts_gb), card_landing
        #   Two cards keep a tier each, a seat a card (▶ qwen4_exp.lisp's tiered cards).
        if self.CARD_GB and (seats > 1 or sockets > 1 or not verbs):
            raise Refused("a card's tier of experts is its one seat's: a seat a card, one CPU worker and the verbs")
        self._plan()
        # a card's experts come from the CPU's memory: a prompt's streamed from their places in RAM, or from the disk tier's
        #   cache, an expert it does not hold read into it first (NN-165, nn's BACKINGS); the tier's promoted from their
        #   places in RAM, or from that cache — the slot the CPU worker holds the expert in when it is made (NN-164, nn's
        #   EXCLUSIVE 2: the tier promotes a layer's most picked, the top of that cache). "auto" keeps the disk's on the CPU
        if self.PROMPT_CARD and self.EXPERTS_FROM not in ("ram", "disk"):
            raise Refused("a prompt's experts on the card come from their places in RAM or the disk's cache there: "
                          "experts_from 'ram' or 'disk', not %r" % self.EXPERTS_FROM)
        if self.CARD_GB and self.EXPERTS_FROM not in ("ram", "disk"):
            raise Refused("a card's tier of experts comes from the experts in RAM or their cache there: experts_from 'ram' or "
                          "'disk', not %r" % self.EXPERTS_FROM)
        # ⭐ THE ROUTED EXPERTS OVER THE SOCKETS (`sockets` 2): a CPU worker a NUMA node (x86_avx2's devices 1 and 2), each
        #   holding in its own node's memory its part of every expert — I/2 of its gate and up rows and H/2 of its down's
        #   rows (▶ nn_load_bundle.expert_part_plan). A layer's experts are then two steps on both sockets at once,
        #   ai_qwen_3's `experts_up` and `experts_down`, each socket's activations copied to the other between them, and
        #   the two halves of the routed sum added on the card (▶ qwen4_exp.lisp `close`). Split by rows, not by inner
        #   width: the down's 640 columns are rotated as 512 and 128, which no even cut of the width keeps whole.
        #   ⛳ The parts are read whole into RAM — the verbs take resident experts only — and a prompt's rows have no
        #   parted form yet, so with two sockets a prompt is read a position at a time.
        if sockets not in (1, 2):
            raise Refused("sockets is 1 or 2, not %r" % (sockets,))
        if sockets == 2:
            if self.EXPERTS_FROM != "ram":
                raise Refused("the experts split over the sockets are held in their memory whole — experts_from 'ram', "
                              "not %r" % self.EXPERTS_FROM)
            if experts_int8:
                raise Refused("the experts split over the sockets are exact: experts_up and experts_down have no integer form")
            if machine is not None and machine.e.devices(machine.cpu) < 3:
                raise Refused("sockets 2 needs two NUMA nodes, a CPU worker each — this machine's x86 family has %d device(s)"
                              % machine.e.devices(machine.cpu))
            self.PPLAN = NL.expert_part_plan(b, 0, self.EXPERTS, sockets)
        self.SOCKETS = sockets
        if (self.CARD_SEATS > 1 or self.CARDS > 1) and sockets > 1:
            raise Refused("seats and cards take the experts on one CPU worker: sockets 1")
        # ⭐ THE SITES AS ai_qwen_4's VERBS (`verbs`, the default): the gated residual's before-step, the attention, the
        #   DeltaNet and the shared expert each one apply, launching the doors their nn words launch in the same order —
        #   the same bits, without the evaluator's work between them. False: the program's nn words, the reference.
        if not isinstance(verbs, bool):
            raise Refused("verbs is true or false, not %r" % (verbs,))
        self.VERBS = verbs
        self.experts_dir = experts_dir or os.path.join(folder.path, "experts")
        self.SLOTS = None
        if self.EXPERTS_FROM == "disk":
            self.SLOTS = min(self.EXPERTS * self.LAYERS,
                             experts_file.cache_bytes(self.EXPERTS * self.LAYERS * self.EPLAN.slot_bytes, experts_ram_gb)
                             // self.EPLAN.slot_bytes)
            if self.SLOTS < 4 * self.TOPK:
                raise Refused("%d experts is too few for a cache of them" % self.SLOTS)
        # the CPU's experts' memory — every expert in RAM, or the cache of them — as the boot's setup reads it before a load
        #   (▶ autotune's `_setup`: where the experts live, from what this process may hold); the load sets it again by its
        #   sockets' plan
        self.cpu_arena = (self.SLOTS if self.SLOTS is not None else self.EXPERTS * self.LAYERS) * self.EPLAN.slot_bytes
        # ⭐ THE NEXT LAYER'S EXPERTS READ AHEAD (NN-138), as qwen3_5.py's: from the disk, each layer's router also runs the
        #   next layer's on its input — its `GUESS` likeliest picks, the likeliest first — and the CPU worker reads the first
        #   of them it does not hold while the layer computes. `experts_ahead`: "auto" as many as the disk starts in that
        #   wait (nn's measure), a number that many, "no" none.
        self.GUESS = 12
        if experts_ahead in ("auto", None):
            self.AHEAD = 0
        elif str(experts_ahead).lower() in ("no", "false", "0"):
            self.AHEAD = None
        elif isinstance(experts_ahead, int) and not isinstance(experts_ahead, bool) and 0 < experts_ahead <= self.GUESS:
            self.AHEAD = experts_ahead
        else:
            raise Refused("experts_ahead is \"auto\", \"no\" or a count of experts from 1 to %d, not %r"
                          % (self.GUESS, experts_ahead))
        if self.SLOTS is None:
            self.AHEAD = None

    # ── what goes where ─────────────────────────────────────────────────────────────────────────────────────────────
    def names_of(self, l):
        n = [s + "." + g for s in SITES for g in GATED] + list(MOE)
        n += list(ATTENTION) if self.kinds[l] == "f" else list(DELTANET)
        return n + (list(NGRAM) if l in self.PLE else [])

    def _plan(self):
        b = self.b
        self.ALL = sorted({n for l in self.run_layers for n in self.names_of(l)})
        self.TYPE = {n: i for i, n in enumerate(self.ALL)}
        first = {n: next(l for l in self.run_layers if n in self.names_of(l)) for n in self.ALL}
        self.PLAN = {n: NL.dense_plan(b, n, first[n]) for n in self.ALL}
        self.COUNT = {n: sum(1 for l in self.run_layers if n in self.names_of(l)) for n in self.ALL}
        for n in self.ALL:                        # a matrix's width one for every layer: the program reads one `mv`
            ds = {b.record(l, n).d for l in self.run_layers if n in self.names_of(l)}
            if len(ds) != 1:
                raise Refused("%s is packed at widths %s across the layers" % (n, sorted(ds)))

        def plan_of(rec, gain=0.0):
            data = np.asarray(rec["vbr_data"]).tobytes()
            lut = np.asarray(rec["vbr_lut"]).astype("<f2").tobytes()
            if gain:                                  # 1 + w where the code is the weight (d = 16), as the loader does
                if rec["d"] != 16:
                    raise Refused("a norm packed at d=%d" % rec["d"])
                data = (np.frombuffer(data, dtype="<f2").astype(np.float32) + gain).astype("<f2").tobytes()
            return data, lut, NL.SlotPlan(len(data), 0, 0, len(lut))
        mixer = "model.language_model.hyper_connection_mixer."
        self.GLOBAL = {"head": plan_of(self.global_["lm_head.weight"])}
        for n in GATED[:3]:
            self.GLOBAL[mixer + n] = plan_of(self.global_[mixer + n], 1.0 if n == "hc_norm.weight" else 0.0)
        self.T_GLOBAL = {k: len(self.ALL) + i for i, k in enumerate(self.GLOBAL)}
        # the card's tier — `card_experts_gb` of slots, or a slot a layer where it is only a prompt's staging — and, for a
        #   prompt's experts on the card, the ring a chunk streams through (two of the largest batches)
        self.TIER_ON = self.PROMPT_CARD or self.CARD_GB > 0
        self.T_CEXP = len(self.ALL) + len(self.GLOBAL) if self.TIER_ON else None
        self.T_STREAM = self.T_CEXP + 1 if self.PROMPT_CARD else None
        self.TIER_SLOTS = max(len(self.run_layers), int(self.CARD_GB * (1 << 30)) // self.EPLAN.slot_bytes)
        self.RING_SLOTS = 2 * expert_tier.STREAM_BATCH_MAX if self.PROMPT_CARD else 0
        self.HEAD_BITS = int(self.global_["lm_head.weight"]["d"])
        self.MIXER_BITS = {n: int(self.global_[mixer + n]["d"]) for n in GATED[1:3]}
        emb = self.global_["model.language_model.embed_tokens.weight"]
        self.emb_data = np.asarray(emb["vbr_data"]).tobytes()
        self.emb_lut = np.asarray(emb["vbr_lut"]).astype("<f2").tobytes()
        self.EMB_ROW_BYTES, self.EMB_BITS, self.EMB_DATA_BYTES = int(emb["row_bytes"]), int(emb["d"]), len(self.emb_data)
        self.TABLE_BYTES = (len(self.emb_data) + len(self.emb_lut) + 255) // 256 * 256
        self.arena = (sum(self.PLAN[n].slot_bytes * self.COUNT[n] for n in self.ALL)
                      + sum(p.slot_bytes for _, _, p in self.GLOBAL.values()))
        H, S, I = self.H, self.S, self.INTER
        # the card's buffers, their bytes (▶ `_buffers`): the pools are made to hold exactly these
        self.BUFS = dict(
            signs=1024, ones512=1024, signs_row=H * 2, zb=S * 2, ones=S * 2, rv=64, res=64, cs=self.ROT * 4,
            sf=S * 4, mix=4 * 24, mix1=4 * 3, mixd=4 * 8, sf2=2 * S * 4, gvf=2 * S * 4, ngr=H * 2, ngrr=H * 2,
            kv=(S + H) * 2, cv=S * 2, cvt=S * 2, cvs=S * 2,
            eu=H * 2, ehu=H * 2, e=H * 2, an=S * 2, anr=S * 2, a1=self.R * 2, a2=self.R * 2, a3=self.R * 2, a3r=self.R * 2,
            gt=S * 2, gs=S * 2, gn=S * 2, m01=H * 2, m012=H * 2, m4=H * 2, m=H * 2, mr=H * 2, m2=H * 2, mr2=H * 2,
            ij=16, ij2=16, ij3=16, inj=16, inj2=16,
            mixed=self.CONV_CH * 2, z=self.VAL * 2, av=self.VH * 2, bv=self.VH * 2, beta=self.VH * 2, sumb=self.VH * 2,
            spb=self.VH * 2, eab=self.VH * 2, prodb=self.VH * 2, gb=self.VH * 2,
            conved=self.CONV_CH * 2,
            core=self.VAL * 2, crot=self.VAL * 2, y=H * 2, y2=H * 2,
            qg=self.QH * 2 * self.AD * 2, qb=self.QH * self.AD * 2, gsig=self.QH * self.AD * 2, kk=self.KVH * self.AD * 2,
            vv=self.KVH * self.AD * 2, scores=self.QH * self.ATTENDED * 4, ao=self.QH * self.AD * 2,
            aog=self.QH * self.AD * 2, aogr=self.QH * self.AD * 2,
            lg=self.EXPERTS * 2, wts=64, lgn=self.EXPERTS * 2, gwts=64, hand=2 * (H + 16 + I), routed=H * 2, routed1=H * 2, sgo=I * 2, suo=I * 2, sact=I * 2,
            sactr=I * 2, sdo=H * 2, segv=16, segs=16, sdow=H * 2, mh=H * 2, mhr=H * 2, logits_h=self.VOCAB * 2)
        C = self.CHUNK                             # a prompt's chunk: a row of each a position (▶ the program's `prompt_rows`)
        self.BUFS.update(sfR=C * S * 4, ngrR=C * H * 2, csR=C * self.ROT * 4, lgR=C * self.EXPERTS * 2,
                         hands=C * self.HAND_ROW, m2R=C * H * 2, mr2R=C * H * 2, inj2R=C * 16, sdowR=C * H * 2,
                         routedR=C * H * 2)
        # ai_qwen_4's scratch: the largest any verb carves of it, every piece on a 256-byte boundary (▶ its layer__abi)
        carve = lambda *halves: sum((2 * n + 255) // 256 * 256 for n in halves)
        QW, KVW, VH, VAL, M, R = self.QH * self.AD, self.KVH * self.AD, self.VH, self.VAL, self.HC, self.R
        IW, IQ, PICK, BLOCKS = self.IH * self.ID + self.ID, self.IH * self.ID, self.PICK, self.max_context // self.RATIO
        self.BUFS["q4s"] = max(carve(S, S, R, R, R, S, M),
                               carve(H, 2 * QW, KVW, KVW, QW, QW, QW, QW, QW, 2 * QW,
                                     IW, IQ, self.ID, self.RATIO * self.ID, 2 * self.ROT, 2 * BLOCKS, 2 * PICK, 2,
                                     PICK * KVW, PICK * KVW),
                               carve(H, self.CONV_CH, VAL, VH, VH, VH, VH, self.CONV_CH, VAL, VAL),
                               carve(I, I, I, I, H, 1, 1))
        # …and a chunk's, its rows at once (the rows verbs, ▶ ai_qwen_4's layer__abi): their scratch, the chunk's attention
        #   scores over the context, a mix a row (no pre, the injection's post, the identity as comb), the sites' rows
        self.BUFS["q4r"] = max(carve(4 * C, C * S, C * S, C * R, C * R, C * R, C * R, C * S, C * S, C * S, M * C * H, C * H,
                                     C * H, C * M, C * M, C * M),
                               carve(4 * C, C * H, C * self.CONV_CH, C * VAL, C * VH, C * VH, C * VH, C * VH, C * self.CONV_CH,
                                     C * VAL, C * VAL),
                               carve(4 * C, 4 * C, C * H, C * 2 * QW, C * KVW, C * QW, C * QW, C * QW, C * QW,
                                     C * IW, C * IQ, C * IQ, (C // self.RATIO + 1) * self.RATIO * self.ID,
                                     (C // self.RATIO + 1) * self.ID, (C // self.RATIO + 1) * 2 * self.ROT, 2 * BLOCKS,
                                     2 * PICK, 2, PICK * KVW, PICK * KVW),
                               carve(4 * C, C * I, C * I, C * I, C * I, C * H, C, C))
        self.MIX_ROW = 2 * M + M * M
        self.BUFS.update(scoresR=C * self.QH * self.ATTENDED * 4, mixR=C * self.MIX_ROW * 4, mR=C * H * 2, yR=C * H * 2,
                         injR=C * 16)
        if self.PROMPT_CARD:                       # the card's `experts_rows` over a chunk: its scratch, as the CPU's
            P = C * self.TOPK
            self.BUFS.update(krscratch=2 * C * H + 16 * P + 2 * P + 8 * P * I + 4 * C * self.TOPK * H + 4096)
        if self.TIER_ON:
            self.BUFS.update(zero_h=H * 2)
        if self.CARD_GB:                           # a position's: the card's experts' scratch, as Qwen 3.5's, its routed sum
            self.BUFS.update(kscratch=2 * (self.TOPK * 2 * I + 2 * self.TOPK * I), routed_k=H * 2)
        self.S_BYTES, self.W_BYTES = self.VH * self.LD * self.LD * 4, self.CONV_CH * 3 * 2
        self.KV_BYTES = self.max_context * self.KVH * self.AD * 2
        self.GVN_BYTES = self.max_context * S * 2
        self.STATE = {}
        for l in self.run_layers:
            if self.kinds[l] == "d":
                self.STATE["s32_%d" % l], self.STATE["cwin%d" % l] = self.S_BYTES, self.W_BYTES
            else:
                self.STATE["kc%d" % l], self.STATE["vc%d" % l] = self.KV_BYTES, self.KV_BYTES
                self.STATE["ik%d" % l] = self.max_context * self.ID * 2                    # the indexer's keys, a position's
                self.STATE["ip%d" % l] = (self.max_context // self.RATIO) * self.ID * 2    # …and a block's
        self.STATE["prev2"] = 8                    # the two tokens before the next position: the n-gram hash reads them
        for l in self.PLE:
            self.STATE["gvn%d" % l] = self.GVN_BYTES
            self.STATE["weng%d" % l] = S * 2
            for k in range(self.CONV_K):
                self.STATE["wtap%d_%d" % (l, k)] = S * 2
        # a seat's own past the first: its conversation's state and what its step carries between its sites
        own = [(name, n) for name, n in self.STATE.items() if not name.startswith(("weng", "wtap"))]
        for seat in range(1, self.CARD_SEATS):
            for name in self.SEAT_BUFS:
                self.BUFS[self.seat_name(name, seat)] = self.BUFS[name]
            for name, n in own:
                self.STATE[self.seat_name(name, seat)] = n

    POOLS = ("partials", "resid", "attn_out", "main", "logits")

    @staticmethod
    def _room(n):
        """An ask's room in a class: on a 256-byte boundary, 64 bytes at least."""
        return max(64, (n + 255) // 256 * 256)

    @classmethod
    def _classes(cls, asks, spare, reach):
        """Five buffer classes' sizes for `asks` (bytes): the five of the asks' own sizes that hold them, and `spare` more
        of each class up to the one `reach` (the largest working buffer) falls in, in the fewest bytes — a class above it
        holds the conversation's state alone, made once, and keeps no spare (▶ `_pools`). ⚖ 2026-11-09 *"let's fix the
        card pool"*: cut at the asks' median, upper quartile, ninetieth percentile and two largest, Flash-Next's 47.8 KB
        of state a position took 119 KB of the card — the n-gram layer's values the largest class and its two spares, the
        indexer's keys rounded into K's class — and past ~200k positions an empty class twice the largest
        (measurements/2026-11-09_auto_context_node03.md ③)."""
        rooms = sorted(set(cls._room(n) for n in asks))
        count = {r: 0 for r in rooms}
        for n in asks:
            count[cls._room(n)] += 1
        top = next((r for r in rooms if r >= cls._room(reach)), rooms[-1])
        m = len(rooms)

        def cost(i, j):                 # a class of rooms[j] holding the asks of rooms (i, j], and its spares
            return rooms[j] * (sum(count[rooms[t]] for t in range(i + 1, j + 1)) + (spare if rooms[j] <= top else 0))
        best = {(1, j): (cost(-1, j), None) for j in range(m)}
        for k in range(2, min(5, m) + 1):
            for j in range(k - 1, m):
                best[(k, j)] = min((best[(k - 1, i)][0] + cost(i, j), i) for i in range(k - 2, j))
        k = min(range(1, min(5, m) + 1), key=lambda k: best[(k, m - 1)][0])   # fewer classes where a spare costs more
        j, sizes = m - 1, []
        while j is not None:
            sizes.append(rooms[j])
            j = best[(k, j)][1]
            k -= 1
        sizes = sorted(sizes)
        while len(sizes) < 5:           # fewer sizes than classes: the rest past the largest, holding nothing
            sizes.append(sizes[-1] * 2)
        return sizes

    @classmethod
    def _pools(cls, sizes, asks, spare, reach):
        """The classes `sizes` holding `asks`, each in the smallest class that fits (▶ nn's `zzprivate_take`), and
        `spare` more of each class up to the one `reach` falls in — none above it, where only the state is."""
        qty = dict.fromkeys(cls.POOLS, 0)
        for n in asks:
            qty[cls.POOLS[next(i for i, s in enumerate(sizes) if s >= n)]] += 1
        top = next(s for s in sizes if s >= cls._room(reach))
        return "".join("nn__buffer_%s__size_bytes:%d\nnn__buffer_%s__qty:%d\n" % (nm, s, nm, qty[nm] + (spare if s <= top else 0))
                       for nm, s in zip(cls.POOLS, sizes))

    def device_params(self):
        """The boot's sections: each card's (its pools, its arena, its collections — every card's the same) and the CPU
        worker's (the experts)."""
        types = "".join("nn__expert__type%d__up_bytes:%d\nnn__expert__type%d__down_bytes:%d\nnn__expert__type%d__slots:%d\n"
                        % (self.TYPE[n], self.PLAN[n].up_bytes, self.TYPE[n], self.PLAN[n].down_bytes, self.TYPE[n],
                           self.COUNT[n]) for n in self.ALL)
        for k, t in self.T_GLOBAL.items():
            p = self.GLOBAL[k][2]
            types += ("nn__expert__type%d__up_bytes:%d\nnn__expert__type%d__down_bytes:%d\nnn__expert__type%d__slots:1\n"
                      % (t, p.up_bytes, t, p.down_bytes, t))
        tier_bytes, n_types = 0, len(self.ALL) + len(self.T_GLOBAL)
        if self.TIER_ON:                           # the tier's slots and the ring's, laid out as the CPU's
            e = self.EPLAN
            for t, n in ((self.T_CEXP, self.TIER_SLOTS), (self.T_STREAM, self.RING_SLOTS)):
                if t is None:
                    continue
                types += ("nn__expert__type%d__up_bytes:%d\nnn__expert__type%d__down_bytes:%d\nnn__expert__type%d__slots:%d\n"
                          % (t, e.up_bytes, t, e.down_bytes, t, n))
                n_types += 1
            tier_bytes = (self.TIER_SLOTS + self.RING_SLOTS) * e.slot_bytes
        n_d = sum(1 for k in self.kinds.values() if k == "d")
        common = ("nn__cartridge_sink__tokens:1\nnn__cartridge_sink__bytes_per_token_layer:64\n"
                  "nn__cartridge_tq4__tokens:1\nnn__cartridge_tq4__bytes_per_token_layer:64\n"
                  "nn__cartridge_tq6__tokens:1\nnn__cartridge_tq6__bytes_per_token_layer:64\n"
                  "nn__cartridge_fp32__tokens:1\nnn__cartridge_fp32__bytes_per_token_layer:64\n"
                  "nn__model__kv_cache_layers_fullattn:%d\nnn__model__kv_cache_layers_deltanet:%d\n"
                  "nn__conversation__text_size_kb:4\nnn__deltanet_state__bytes_per_layer:64\n"
                  "nn__model__layer_types:%s\n" % (self.LAYERS - n_d, n_d, "".join(self.kinds[l] for l in range(self.LAYERS))))
        # ⛔ THE CARD'S AND THE CPU'S CLASSES ARE THE SAME SIZES, NAME FOR NAME: a copy between the two reads the source
        #   buffer's room from its class in the copying worker's own table (▶ nn's `room_of_any`), so a class sized
        #   otherwise on the other side misreports a crossing buffer's room. `MEASURED` node03 2026-10-31: with the CPU's
        #   classes Qwen 3.5's, a 256-row chunk's hands and routed rows refused every copy over 1 MiB — the card's
        #   "resid" 1 MiB read against the CPU's 2 MB (measurements/2026-10-31_nn141_flash_next_node03.md). The CPU's
        #   largest class alone is raised, to its rows' scratch; nothing of that class crosses.
        card_asks = list(self.BUFS.values()) + list(self.STATE.values())
        self.CLASSES = self._classes(card_asks, 2, max(self.BUFS.values()))
        card = (self._pools(self.CLASSES, card_asks, 2, max(self.BUFS.values()))
                + "nn__expert_cache_l1__size_mb:%d\n" % ((self.arena + tier_bytes + (256 << 20)) >> 20) + common
                + "nn__expert__types:%d\n%s" % (n_types, types)
                + "nn__ram_buffer_result__size_bytes:64\nnn__ram_buffer_result__qty:1\n"
                  "nn__ram_buffer_experts__size_bytes:64\nnn__ram_buffer_experts__qty:1\n"
                  "nn__ram_buffer_table__size_bytes:%d\nnn__ram_buffer_table__qty:1\n" % self.TABLE_BYTES)
        p = self.EPLAN if self.SOCKETS == 1 else self.PPLAN
        slots = self.SLOTS if self.SLOTS is not None else self.EXPERTS * self.LAYERS
        self.cpu_arena = slots * p.slot_bytes
        # a prompt's on the CPU: its hand rows, its routed rows and `experts_rows`' scratch (▶ ai_qwen_3's `experts_rows`,
        #   sized as qwen3_5.py sizes it), in the largest of the pools
        P, H, I = self.CHUNK * self.TOPK, self.H, self.INTER
        self.RSCRATCH = 2 * self.CHUNK * H + 16 * P + 2 * P + 8 * P * I + 4 * self.CHUNK * self.TOPK * H + 64
        if self.SOCKETS == 1:
            cpu_asks = [1024, H * 2, 2 * (H + 16), H * 2, 131072, self.CHUNK * self.HAND_ROW, self.CHUNK * H * 2,
                        self.RSCRATCH]
        else:                                      # a socket's: the hand, its activations and the other's, its half sum
            cpu_asks = [1024, H * 2, 2 * (H + 16 + I), self.ACT_BYTES, self.ACT_BYTES, H * 2, 131072]
        cpu_asks += [2 * (H + 16), H * 2] * (self.CARD_SEATS - 1)     # a seat's own hand and routed sum past the first
        cpu_asks += [2 * (H + 16), H * 2, self.CHUNK * self.HAND_ROW, self.CHUNK * H * 2] * (self.CARDS - 1)   # a card's own
        cpu_classes = self.CLASSES[:-1] + [max(self.CLASSES[-1], (max(cpu_asks) + 255) // 256 * 256)]
        cpu = (self._pools(cpu_classes, cpu_asks, 6, max(cpu_asks))
               + "nn__expert_cache_l1__size_mb:%d\n" % ((self.cpu_arena + (256 << 20)) >> 20) + common
               + "nn__expert__types:1\nnn__expert__type0__up_bytes:%d\nnn__expert__type0__down_bytes:%d\n"
                 "nn__expert__type0__slots:%d\n" % (p.up_bytes, p.down_bytes, slots))
        return dict([(k, card) for k in range(self.CARDS)] + [(self.CPU + s, cpu) for s in range(self.SOCKETS)])

    @property
    def ACT_BYTES(self):
        """A socket's activations: every pick's I halves, its own part in their places and zeros elsewhere."""
        return 2 * self.TOPK * self.INTER

    def workers(self):
        """The cards, then the CPU: the whole machine's pool, or with `sockets` 2 a worker a NUMA node."""
        cpus = [(self.m.cpu, 0)] if self.SOCKETS == 1 else [(self.m.cpu, 1 + s) for s in range(self.SOCKETS)]
        return [(self.m.silicon, c) for c in self.CARD_IDS] + cpus

    # ── the load ────────────────────────────────────────────────────────────────────────────────────────────────────
    def load(self, progress=None):
        m = self.m
        for k in reversed(range(self.CARDS)):
            m.act_as(k)
            self._dense(progress if k == 0 else None)
        self.GLOBAL = {k: v[2] for k, v in self.GLOBAL.items()}       # their bytes the engine's now
        self._cpu_experts(progress)
        self._bindings()
        self._ngram_table()
        # ⭐ A SEAT'S NAMES ARE THE PROGRAM'S OWN: the last card's bound first and the programs defined over them, then
        #   each card's seats' frozen in the views their steps run in — a view is a snapshot, which nothing bound after it
        #   reaches (▶ sys's `bindings__viewonly`): a seat past the first has its own names bound over its card's for its
        #   two views and left again. `names` is the view the CPU computes a seat's experts in, `names_s<seat>` the one
        #   the seat's steps run in — which holds that `names` — and card 0's seat 0's are made last, so the bindings as
        #   they stand, and `names`, are seat 0's (▶ qwen4_exp_cpu.lisp)
        self.AT_OF = {}
        for k in reversed(range(self.CARDS)):
            self._card_names(k)
            if k == self.CARDS - 1:
                m.define_programs(self._procedures)
            for seat in reversed(range(self.CARD_SEATS)):
                if seat:
                    self._seat_names(seat, True)
                m.view("names")
                m.view(self.view_of(k * self.CARD_SEATS + seat))
                if seat:
                    self._seat_names(seat, False)
        self.AT = self.AT_OF[CARD]
        self.emb_data = self.emb_lut = None
        self.b.release()                            # every layer's file let go (▶ `Bundle.release`): its pages are
                                                    # the experts' bytes again, copied out of it at the load

    def _dense(self, progress):
        """The acting card's whole dense model: every layer's matrices and the head's and mixer's."""
        m, b = self.m, self.b
        loader = m.loader()
        for i, l in enumerate(self.run_layers):
            for n in self.names_of(l):
                loader.layer_width(l, self.TYPE[n], 1)
                loader.load_dense(b, self.PLAN[n], self.TYPE[n], n, l)
            if progress:
                progress(i + 1, len(self.run_layers))
        for k, t in self.T_GLOBAL.items():
            data, lut, p = self.GLOBAL[k]
            loader.layer_width(0, t, 1)
            loader.load_bytes(p, t, 0, data, lut)
        if self.TIER_ON:                           # the tier's and the ring's bands a layer, empty at boot
            for l in self.run_layers:
                loader.layer_width(l, self.T_CEXP, self.EXPERTS)
                if self.T_STREAM is not None:
                    loader.layer_width(l, self.T_STREAM, self.EXPERTS)
        if loader.handed() != 0:
            raise Refused("%d slots were reserved and never published" % loader.handed())

    def _ngram_table(self):
        """⭐ THE N-GRAM ROWS IN THE PROGRAM (▶ nn's `nn__ngram__rows`): the table's file opened once, every card's, and
        `NGT` — where its rows lie, the hash's multipliers, sizes and offsets, the end of text that stops the look back and
        what a place past it reads as — so a position reads its own rows and a run of tokens is one program (`generate`).
        A prompt's chunk keeps the host's (`prefill`, a chunk at a time)."""
        ng, path = self.ng, os.path.join(self.folder.path, "ngram", "rows.e4m3")
        if set(path) & set(" ()'\"\t"):
            raise Refused("the n-gram table %r holds a character a program cannot name" % path)
        kind, handle = self.m.run("(sys__file__open '%s)" % path)
        if kind != self.m.K["sys__value_int"] or handle == 0:
            raise Refused("the n-gram table %s could not be opened" % path)
        self.m.table("NGT", [handle, 0, ng.width, 0, 0, ng.N, ng.per, ng.eos, ng.eos, 0, repr(float(ng.scale))]
                     + [int(v) for v in ng.mult] + [int(v) for v in ng.sizes.reshape(-1)]
                     + [int(v) for v in ng.offsets.reshape(-1)])

    def _card_names(self, k):
        """Card `k`'s names, bound under the program's: its embedding, its buffers and tables, its hand-off's buffers on
        the CPU and its values."""
        m = self.m
        m.act_as(k)
        self.AT = {}
        self.AT["EMBEDDING"] = m.buffer(self.TABLE_BYTES, "EMBEDDING", ram=True)
        m.write(self.AT["EMBEDDING"], self.emb_data)
        m.write(self.AT["EMBEDDING"] + self.EMB_DATA_BYTES, self.emb_lut)
        self._buffers()
        self._tables()
        self._answer_buffers()                     # `generate`'s OUT and STOP (▶ generate.py)
        if self.TIER_ON:
            self._card_tier(k)
        if self.SOCKETS == 1:
            self._cpu_hand_off()
        m.act_as(k)
        hp = self.GLOBAL["head"]
        plane = lambda at, n: "(nn__expert__plane 0 %d 0 %d %d)" % (self.T_GLOBAL["head"], at, n)
        m.bind_values(dict(CARD=k, HEAD_CODES=plane(hp.at_up_data, hp.up_data), HEAD_SCALES=plane(hp.at_down_lut, hp.down_lut)))
        self.AT_OF[k] = self.AT
        m.act_as(CARD)

    def _card_tier(self, k):
        """Card `k`'s tier (`card_experts_gb`, or `prompt_on` "card"'s staging): each layer's experts tables over the card's
        slots — a position's with its scratch (`exk`), a chunk's with the chunk's (`exkr`) — the tier fed from the CPU's
        memory, streaming through its ring where a prompt's experts are the card's, and the binding."""
        m, e = self.m, self.EPLAN
        for l in self.run_layers:
            cells = [l, self.T_CEXP, e.at_up_lut, e.at_down_data, e.at_down_lut, self.H, self.INTER, self.EXPERTS, self.TOPK,
                     int(self.b.record(l, GU).d), 0, 0, 0, 0, 0, 0]
            if self.PROMPT_CARD:
                m.table("exkr%d" % l, ["signs", "krscratch", "zero_h"] + cells + [0])
            if self.CARD_GB:
                m.table("exk%d" % l, ["signs", "kscratch", "zero_h"] + cells)
        if self.PROMPT_CARD:
            m.array("EXPERTS_CARD_ROWS", ["exkr%d" % l for l in self.run_layers])
        if self.CARD_GB:
            m.array("EXPERTS_TIER", ["exk%d" % l for l in self.run_layers])
        disk = self.EXPERTS_FROM == "disk"
        expert_tier.tier(m, "tier_card", self.T_CEXP, self.EXPERTS, self.TOPK, self.H, self.INTER, e, self.LAYERS,
                         None if disk else {l: "src%d" % l for l in self.run_layers}, landing=self.CARD_LANDING,
                         stream=self.T_STREAM, slots=dict(worker=self.CPU, type=0) if disk else None,
                         # ⭐ A PROMPT ON THE CARD FROM THE DISK (NN-165): an expert a chunk picks that the cache does not
                         #   hold read into it from the layer's file, a batch ahead of its copy to the ring
                         backings={l: self._backing[(self.CPU, l)] for l in self.run_layers}
                         if disk and self.T_STREAM is not None else None)
        expert_tier.bind(m, {k: "tier_card"}, len(self.workers()))
        # past any chunk's picks of one expert: a prompt's experts through the ring, none promoted by it — the tier is the
        #   positions' to fill
        m.must("(sys__add_bindings %d '%s %d)" % (m.env, expert_tier.THRESHOLD, self.CHUNK + 1), "the promotion threshold",
               "sys__value_true")

    def _buffers(self):
        m, H, S = self.m, self.H, self.S
        g = self.global_
        signs = 1.0 - 2.0 * np.unpackbits(np.frombuffer(g["__signs__"]["plane"], dtype=np.uint8),
                                          bitorder="little").astype(np.float64)
        given = dict(signs=_half(signs[:512]), ones512=_half(np.ones(512)), signs_row=_half(np.tile(signs[:512], H // 512)),
                     zb=bytes(S * 2), ones=_half(np.ones(S)),
                     mix=_f32(np.concatenate([np.zeros(2 * self.HC), np.eye(self.HC).reshape(-1)])),
                     mixR=_f32(np.tile(np.concatenate([np.zeros(2 * self.HC), np.eye(self.HC).reshape(-1)]), self.CHUNK)),
                     mix1=_f32([0, 1, 1]), mixd=_f32([0, 0, 0, 0, 1, 0, -1, 0]),   # two streams: out_0 = stream_0 - stream_1
                     zero_h=bytes(H * 2))
        for name, n in self.BUFS.items():
            self.AT[name] = m.buffer(n, name, given.get(name))
        for name, n in self.STATE.items():
            if name.startswith(("weng", "wtap")):
                continue
            prev2 = name.split("_s")[0] == "prev2"
            self.AT[name] = m.buffer(n, name, self.EOS2 if prev2 else None)
            if not prev2:
                m.zero(self.AT[name], n)
        for l in self.PLE:                         # the n-gram gate's two norms folded as DeepSeek's runtime folds its q and k
            dec = lambda n: NU.decode_record(*self._layer_record(l, n)).astype(np.float64)
            self.AT["weng%d" % l] = m.buffer(S * 2, "weng%d" % l,
                                            _half((1 + dec("ple.norm_query.weight")) * (1 + dec("ple.norm_key.weight"))))
            taps = dec("ple.conv1d.weight")[:, 0, :]
            for k in range(self.CONV_K):
                self.AT["wtap%d_%d" % (l, k)] = m.buffer(S * 2, "wtap%d_%d" % (l, k), _half(taps[:, k]))
        for seat in range(self.CARD_SEATS):
            m.node_array(self.seat_name("picks", seat), self.TOPK)
        m.node_array("guess", self.GUESS)
        m.node_array("toks", self.CHUNK)          # a prompt's chunk's tokens, set by `prefill` before the chunk runs
        for seat in range(self.CARD_SEATS):        # what a seat's steps carry (▶ qwen4_exp.lisp's chain): layer, position, end
            m.node_array(self.seat_name("HERE", seat), 3)

    def _layer_record(self, l, n):
        r = self.b.record(l, n)
        return r.rec, self.b.blob(l)

    def plane(self, l, n, which):
        p = self.PLAN[n]
        at, size = (p.at_up_data, p.up_data) if which == "data" else (p.at_down_lut, p.down_lut)
        return "(nn__expert__plane %d %d 0 %d %d)" % (l, self.TYPE[n], at, size)

    def gplane(self, k, which):
        p = self.GLOBAL[k]
        at, size = (p.at_up_data, p.up_data) if which == "data" else (p.at_down_lut, p.down_lut)
        return "(nn__expert__plane 0 %d 0 %d %d)" % (self.T_GLOBAL[k], at, size)

    def _tables(self):
        m, b = self.m, self.b
        for l in self.run_layers:
            have = set(self.names_of(l))
            cells = []
            for key, name, w in CELLS:
                if key == "WENG":
                    cells += ["weng%d" % l if l in self.PLE else 0]
                elif key == "TAP":
                    cells += ["wtap%d_%d" % (l, k) if l in self.PLE else 0 for k in range(self.CONV_K)]
                elif name not in have:
                    cells += [0] * w
                elif w == MATRIX:
                    cells += [self.plane(l, name, "data"), self.plane(l, name, "lut"), int(b.record(l, name).d)]
                else:
                    cells += [self.plane(l, name, "data")]
            m.table("lt%d" % l, cells)
        mixer = "model.language_model.hyper_connection_mixer."
        head = [0] * TABLE_CELLS
        head[AT["AN"]] = self.gplane(mixer + "hc_norm.weight", "data")
        for key, n in (("AD", "input_mix_weight_down.weight"), ("AU", "input_mix_weight_up.weight")):
            head[AT[key]:AT[key] + 3] = [self.gplane(mixer + n, "data"), self.gplane(mixer + n, "lut"), self.MIXER_BITS[n]]
        m.table("HEADT", head)
        # the shape ai_qwen_4's verbs read (▶ its contracts/objects/layer.cuh): the buffers, the integers, the epsilon
        shape = [self.H, self.HC, self.R, self.KH, self.VH, self.LD, self.QH, self.KVH, self.AD, self.ROT, self.INTER,
                 self.EXPERTS, self.HAND_ROW]
        qsa = [self.THETA, self.IH, self.ID, self.RATIO, self.BUDGET]
        m.table("Q4", ["signs", "ones", "q4s", "cs", "scores", "mix"] + shape + [1] + qsa + [repr(float(self.EPS))])
        m.table("Q4R", ["signs", "ones", "q4r", "csR", "scoresR", "mixR"] + shape + [self.CHUNK] + qsa + [repr(float(self.EPS))])
        for seat in range(1, self.CARD_SEATS):           # a seat's own position shape: its angles
            m.table(self.seat_name("Q4", seat), ["signs", "ones", "q4s", self.seat_name("cs", seat), "scores", "mix"] + shape
                    + [1] + qsa + [repr(float(self.EPS))])
        m.array("LT", ["lt%d" % l for l in self.run_layers])
        m.array("ATTN", [self.kinds[l] == "f" for l in self.run_layers])
        m.array("NGRAM_AT", [l in self.PLE for l in self.run_layers])
        # the next layer's place in RUN, where its experts are read ahead (`experts_ahead`); 0 where none — the last
        m.array("AHEAD_NEXT", [i + 1 if self.AHEAD is not None and i + 1 < len(self.run_layers) else 0
                               for i in range(len(self.run_layers))])
        for arr, fmt, kind in (("S32S", "s32_%d", "d"), ("CWINS", "cwin%d", "d"), ("KCS", "kc%d", "f"), ("VCS", "vc%d", "f"),
                               ("IKS", "ik%d", "f"), ("IPS", "ip%d", "f")):
            for seat in range(self.CARD_SEATS):
                m.array(self.seat_name(arr, seat),
                        [self.seat_name(fmt % l, seat) if self.kinds[l] == kind else 0 for l in self.run_layers])
        for seat in range(self.CARD_SEATS):
            m.array(self.seat_name("GVNS", seat),
                    [self.seat_name("gvn%d" % l, seat) if l in self.PLE else 0 for l in self.run_layers])

    def _expert_pieces(self, view, l, x, part):
        """Expert `x` of layer `l` as its slot's four pieces (▶ `experts_file`), the runs of the layer `view` it lies in."""
        b, p, out = self.b, self.EPLAN, []
        for rec, data_into, lut_into in ((b.record(l, GU), p.at_up_data, p.at_up_lut),
                                         (b.record(l, DN), p.at_down_data, p.at_down_lut)):
            (d, nd), (s, ns) = rec.expert_spans(x, self.EXPERTS)
            out += [(data_into, view[d:d + nd]), (lut_into, view[s:s + ns])]
        return out

    def _cpu_experts(self, progress):
        """The CPU worker: every layer's experts in its memory, mapped, or a cache of them read from their file; the
        scratch every card's hand-off shares (▶ `_cpu_hand_off` for a card's own) and each layer's experts table
        (ai_qwen_3's cells)."""
        m, H, e, CPU = self.m, self.H, self.EPLAN, self.CPU
        layers = self.run_layers
        if self.SOCKETS > 1:
            return self._socket_experts(progress)
        backing = self._backing = {}
        if self.EXPERTS_FROM == "disk":
            backing = self._backing = experts_file.cache_experts(m, self.b, [(CPU, 0, 1)], e, layers, self.EXPERTS,
                                                                 self.experts_dir, self._expert_pieces, layers,
                                                                 progress=progress)
        elif self.EXPERTS_FROM == "mapped":
            experts_file.map_experts(m, self.b, [(CPU, 0, 1)], e, layers, self.EXPERTS, self.experts_dir,
                                     self._expert_pieces, layers, progress=progress)
        else:
            m.act_as(CPU)
            loader = m.loader()
            self.SRC = {}                          # every expert's address in the CPU's memory: what a card's tier copies
            for i, l in enumerate(layers):
                loader.layer_width(l, 0, self.EXPERTS)
                self.SRC[l] = [loader.load_expert(self.b, e, 0, l, x, self.EXPERTS) for x in range(self.EXPERTS)]
                if progress:                       # the experts' load said a layer at a time, as the dense part's
                    progress(i + 1, len(layers))
            if loader.handed() != 0:
                raise Refused("%d expert slots were reserved and never published" % loader.handed())
        m.act_as(CPU)
        signs = 1.0 - 2.0 * np.unpackbits(np.frombuffer(self.global_["__signs__"]["plane"], dtype=np.uint8),
                                          bitorder="little").astype(np.float64)
        m.buffer(1024, "c_signs", _half(signs[:512]))
        m.buffer(H * 2, "c_zero", _half(np.zeros(H)))
        m.buffer(131072, "c_scratch")
        m.buffer(self.RSCRATCH, "c_rscratch")
        for l in layers:
            cells = ["c_signs", "c_scratch", "c_zero", l, 0, e.at_up_lut, e.at_down_data, e.at_down_lut,
                     H, self.INTER, self.EXPERTS, self.TOPK, int(self.b.record(l, GU).d), backing.get((CPU, l), 0)]
            m.table("exc%d" % l, cells + ([0] * (19 - len(cells)) + [1] if self.EXPERTS_INT8 else []))
            # a prompt's: the chunk's scratch in place of a position's, and its products in the position's arithmetic —
            #   exact unless `experts_int8` (ai_qwen_3's AI_QWEN_3__EXP__INT8, here always given: absent, a prompt's rows
            #   would be in integers whatever a position's are)
            rows = ["c_rscratch" if c == "c_scratch" else c for c in cells]
            m.table("exrc%d" % l, rows + [0] * (19 - len(rows)) + [1 if self.EXPERTS_INT8 else 0])
        m.array("EXPERTS_CPU", ["exc%d" % l for l in layers])
        m.array("EXPERTS_CPU_ROWS", ["exrc%d" % l for l in layers])
        if self.TIER_ON and self.EXPERTS_FROM != "disk":     # from the disk the tier reads the cache's slots (NN-164)
            for l in layers:
                m.array("src%d" % l, self.SRC[l])
        m.act_as(CARD)

    def _cpu_hand_off(self):
        """A card's own buffers on the CPU worker, under the program's names (▶ `_card_names`): the hand copied across and
        the routed sum, a seat's each, and a prompt chunk's hand rows and routed rows. The CPU computes one card's at a
        time, so the experts' scratch is every card's."""
        m, H = self.m, self.H
        m.act_as(self.CPU)
        m.buffer(2 * (H + 16), "c_hand")
        m.buffer(H * 2, "c_routed")
        for seat in range(1, self.CARD_SEATS):         # a seat's own hand and routed sum past the first (▶ its view)
            m.buffer(2 * (H + 16), self.seat_name("c_hand", seat))
            m.buffer(H * 2, self.seat_name("c_routed", seat))
        m.buffer(self.CHUNK * self.HAND_ROW, "c_hands")
        m.buffer(self.CHUNK * H * 2, "c_rrows")

    def _socket_experts(self, progress):
        """Each socket's part of every layer's experts in its own memory (`sockets` 2), a layer at a time for both so a
        layer's file is read once; its buffers and each layer's experts table, the part's cells after the fixed ones."""
        m, H, p, P, CPU = self.m, self.H, self.PPLAN, self.SOCKETS, self.CPU
        loaders = {}
        for s in range(P):
            m.act_as(CPU + s)
            loaders[s] = m.loader()
        for i, l in enumerate(self.run_layers):
            for s in range(P):
                m.act_as(CPU + s)
                loaders[s].layer_width(l, 0, self.EXPERTS)
                for x in range(self.EXPERTS):
                    loaders[s].load_expert_part(self.b, p, 0, l, x, s, P, self.EXPERTS)
            if progress:
                progress(i + 1, len(self.run_layers))
        signs = 1.0 - 2.0 * np.unpackbits(np.frombuffer(self.global_["__signs__"]["plane"], dtype=np.uint8),
                                          bitorder="little").astype(np.float64)
        for s in range(P):
            m.act_as(CPU + s)
            if loaders[s].handed() != 0:
                raise Refused("%d expert slots were reserved and never published on socket %d" % (loaders[s].handed(), s))
            m.buffer(1024, "c_signs%d" % s, _half(signs[:512]))
            m.buffer(H * 2, "c_zero%d" % s, _half(np.zeros(H)))
            m.buffer(2 * (H + 16 + self.INTER), "c_hand%d" % s)
            m.buffer(self.ACT_BYTES, "c_act%d" % s)
            m.buffer(self.ACT_BYTES, "c_other%d" % s)
            m.buffer(H * 2, "c_routed%d" % s)
            m.buffer(131072, "c_scratch%d" % s)
            for l in self.run_layers:
                m.table("exc%d_%d" % (l, s), ["c_signs%d" % s, "c_scratch%d" % s, "c_zero%d" % s, l, 0, p.at_up_lut,
                                              p.at_down_data, p.at_down_lut, H, self.INTER, self.EXPERTS, self.TOPK,
                                              int(self.b.record(l, GU).d), 0, s, P])
            m.array("EXS%d" % s, ["exc%d_%d" % (l, s) for l in self.run_layers])
        for name in ("EXS", "C_HAND", "C_ACT", "C_OTHER", "C_ROUTED"):
            base = name if name == "EXS" else "c_" + name[2:].lower()
            m.array(name, ["%s%d" % (base, s) for s in range(P)])
        m.act_as(CARD)
        m.array("SOCKET", [CPU + s for s in range(P)])

    def _bindings(self):
        """The values every card's program reads alike — a card's own, CARD and its head's planes, are its names'
        (▶ `_card_names`)."""
        H, S = self.H, self.S
        values = dict(HIDDEN=H, STREAMS=self.HC, SW=S, RANK=self.R, KH=self.KH, VH=self.VH,
                      LD=self.LD, KEY=self.KEY, VAL=self.VAL, CONV_CH=self.CONV_CH,
                      QH=self.QH, KVH=self.KVH, AD=self.AD, QW=self.QH * self.AD, KVW=self.KVH * self.AD, ROT=self.ROT,
                      THETA=self.THETA, NE=self.EXPERTS, TOPK=self.TOPK, INTER=self.INTER, VOCAB=self.VOCAB, EPS=self.EPS,
                      RUN=len(self.run_layers), CONV_K=self.CONV_K, DIL=self.DIL, HIDDEN_BYTES=2 * H,
                      HAND_BYTES=2 * (H + 16), CPU=self.CPU, CARDS=self.CARDS, SOCKETS=self.SOCKETS, ACT_BYTES=self.ACT_BYTES,
                      VERBS=self.VERBS, SEATS=self.CARD_SEATS, PROMPT_CARD=self.PROMPT_CARD, CARD_TIER=self.CARD_GB > 0,
                      STOPS=STOPS_MOST,
                      NOTE_STREAM=expert_tier.NOTE_STREAM,
                      QUARTER=1.0 / self.HC, TWO=2.0, NEG=-1.0, GATE_SIGMOID=1,
                      EMB_HALVES=self.EMB_ROW_BYTES // 2, EMB_SCALES=self.EMB_DATA_BYTES // 2, EMB_BITS=self.EMB_BITS,
                      HEAD_BITS=self.HEAD_BITS, GUESS=self.GUESS, AHEAD=self.AHEAD or 0,
                      HAND_ROW=self.HAND_ROW, HAND_HALVES=self.HAND_ROW // 2, HAND_PICKS_AT=self.HAND_ROW - 8 * self.TOPK,
                      HAND_W_AT=2 * H)
        values.update({"C_" + k: v for k, v in AT.items()})
        self.m.bind_values(values)
        # each layer's front, the procedure the program chooses for it when it loads (▶ qwen4_exp.lisp's end) — every
        #   card's alike, so bound once, under every card's names
        for name in ("FRONTS", "PLAIN_FRONTS"):
            self.m.node_array(name, len(self.run_layers))

    def _procedures(self):
        """The program: `lisp/qwen4_exp.lisp`, the pictures the card hands the CPU, and `generate` (▶
        `Machine.define_programs`)."""
        self.m.worker_moved = {1: self.CPU}               # the file's worker 1 the CPU, after the cards
        self.m.load_source(self.folder.program("qwen4_exp"))
        self.m.load_source(self.folder.program("qwen4_exp_cpu"))
        self.m.load_source(self.folder.program("generate"))

    # ── a conversation's state ──────────────────────────────────────────────────────────────────────────────────────
    active_lora = None                             # no adapter: the session's `_activate` loads none

    def state_regions(self, length, where=None):
        """(address, bytes) of everything a conversation of `length` positions has written: each DeltaNet layer's state
        and window whole, each attention layer's key and value rows `0..length` and its indexer's — a position's key `0..length`,
        a block's for the blocks finished in them — the n-gram layer's normalised gated
        values `0..length` (its convolution reads back over them), and the two tokens the hash reads next. `where`: each
        buffer's address — card 0's by default; a table of zeros answers the sizes before the boot."""
        at = self.AT if where is None else where
        rows = dict(kc=self.KVH * self.AD * 2, vc=self.KVH * self.AD * 2, ik=self.ID * 2, gvn=self.S * 2)
        out = []
        for name, n in self._own_state():
            key = next((k for k in rows if name.startswith(k)), None)
            if name.startswith("ip"):                # the indexer's block keys: the blocks `length` positions finished
                out.append((at[name], (length // self.RATIO) * self.ID * 2))
            else:
                out.append((at[name], length * rows[key] if key else n))
        return out

    def _own_state(self):
        """(name, bytes) of a conversation's state, a card's seat 0's names (a seat's are `seat_name` of them)."""
        return [(name, n) for name, n in self.STATE.items()
                if not name.startswith(("weng", "wtap"))
                and not any(name.endswith("_s%d" % k) for k in range(1, self.CARD_SEATS))]

    def seat_state_regions(self, length, seat):
        """`state_regions` of seat `seat`'s conversation: its own buffers, on its card."""
        card, local = self.place(seat)
        return self.state_regions(length, where={name: self.AT_OF[card][self.seat_name(name, local)]
                                                 for name, _ in self._own_state()})

    # ── seats ───────────────────────────────────────────────────────────────────────────────────────────────────────
    SEATS_MOST = 4
    PARKS = True                                   # a conversation's state reads off the card and back (▶ the session)
    # ⭐ EVERY SEAT STEPS ON ITS OWN (⚖ 2026-11-04 the resident machine, qwen4_exp.lisp's chain): a seat's position and
    #   prompt run on its card's block in its own view (`step`, `prefill`), so the session drives each seat from a thread
    #   of its own
    ALONE = True

    @property
    def SEAT(self):
        """The seat `on_seat` chose on this thread, 0 where it chose none: each seat's thread steps its own."""
        return getattr(self.__dict__.setdefault("_seat_thread", threading.local()), "seat", 0)

    @SEAT.setter
    def SEAT(self, seat):
        self.__dict__.setdefault("_seat_thread", threading.local()).seat = seat
    # what a seat's view binds (▶ `load`): the names its programs read a conversation by, what its steps carry, the
    #   logits its head leaves, and its hand and routed sum on the CPU
    SEAT_NAMES = ("sf", "hand", "picks", "m2", "mr2", "inj2", "sdow", "ngr", "prev2", "cs", "Q4", "S32S", "CWINS", "KCS", "VCS",
                  "IKS", "IPS",
                  "GVNS", "HERE", "logits_h", "c_hand", "c_routed")
    SEAT_BUFS = ("sf", "hand", "m2", "mr2", "inj2", "sdow", "ngr", "cs", "logits_h")

    @staticmethod
    def seat_name(name, seat):
        """A card's seat `seat`'s own name for a buffer, a table or an array: seat 0's is the name itself."""
        return name if seat == 0 else "%s_s%d" % (name, seat)

    def place(self, seat):
        """Seat `seat`'s card and its seat on that card."""
        return divmod(seat, self.CARD_SEATS)

    def view_of(self, seat):
        """The view seat `seat`'s steps run in (▶ `load`)."""
        return "names_s%d" % seat

    def _seat_names(self, seat, enter):
        """A card's seat `seat`'s own names bound over the program's — or left again — for its views (▶ `load`)."""
        m = self.m
        if enter:
            text = " ".join("(sys__add_bindings %d '%s %s)" % (m.env, n, self.seat_name(n, seat)) for n in self.SEAT_NAMES)
        else:
            text = " ".join("(sys__remove_bindings %d '%s)" % (m.env, n) for n in self.SEAT_NAMES)
        m.must("(begin %s (< 0 1))" % text, "seat %d's names" % seat, "sys__value_true")

    def _at(self, name, seat=None):
        """The address of seat `seat`'s own `name` — the seat `on_seat` chose's by default."""
        card, local = self.place(self.SEAT if seat is None else seat)
        return self.AT_OF[card][self.seat_name(name, local)]

    @contextlib.contextmanager
    def on_seat(self, seat):
        """What runs inside runs on seat `seat`'s conversation — its reset, its prompt, its steps — and on seat 0 after."""
        if not 0 <= seat < self.SEATS:
            raise Refused("seat %r of %d" % (seat, self.SEATS))
        was, self.SEAT = self.SEAT, seat
        try:
            yield self
        finally:
            self.SEAT = was

    # ── a position ──────────────────────────────────────────────────────────────────────────────────────────────────
    def reset(self):
        """A new conversation on the seat `on_seat` chose: every state buffer zeroed, the tokens before position 0 the
        hash's end of text."""
        for name, n in self._own_state():
            self.m.zero(self._at(name), n)
        self.m.write(self._at("prev2"), self.EOS2)

    @staticmethod
    def _end(greedy, head):
        """A card section's end (▶ qwen4_exp.lisp `ends`): 1 the greedy head, 2 the logits left, 0 none."""
        return (1 if greedy else 2) if head else 0

    def step(self, token, pos, greedy=True, sample=None, lag=0, head=True):
        """One position on the seat `on_seat` chose, on its card's block in its view: the token in, the next greedy token
        out — or the logits left (`greedy` false) for a host's sampler, or nothing (`head` false: a prompt's inner
        positions). The program reads the token's n-gram rows itself (▶ `_ngram_table`), from the two before it in
        `prev2`, the conversation's state on the card."""
        if pos >= self.max_context:
            raise Refused("position %d is past the context of %d" % (pos, self.max_context))
        if sample is not None:
            raise Refused("this program's head is the greedy one or the logits: no sampler on the card yet")
        card, e = self.place(self.SEAT)[0], self._end(greedy, head)
        # ⭐ A CHAIN OF STEPS (▶ qwen4_exp.lisp's chain), the last answering; the experts over two sockets a position whole
        text = ("(chain_position %d %d %d)" % (token, pos, e) if self.SOCKETS == 1
                else "(begin (position %d %d) (ends %d))" % (token, pos, e))
        return self.m.run_in(card, self._inline(text), view=self.view_of(self.SEAT))[1]

    def _inline(self, text):
        """A seat's program run inline (NN-159): it keeps its card's scratch between its verbs, so with seats no slice gives
        its block to another seat's step in the middle of it, nor does a wait park it there (▶ sys's
        `sys__scheduler__compute_inline`). One seat runs as it is."""
        return "(sys__scheduler__compute_inline '%s)" % text if self.SEATS > 1 else text

    def prefill(self, tokens, first, greedy=True, sample=None, rows=True, **_):
        """Positions `first ..` for `tokens`, the next token from the last: a CHUNK at a time as rows (the program's
        `prompt_rows`: each layer over the chunk before the next, the routed experts once over its rows — on the card from
        `card_from` rows where `prompt_on` has them there, else on the CPU), or with `rows` false — or the experts over two
        sockets, whose parts have no rows form yet — a position at a time. The host's part of a chunk first: its n-gram
        rows into `ngrR` — the hash's two tokens before it from `prev2`, which the chunk's last two then replace — and its
        tokens into `toks`."""
        tokens = [int(t) for t in tokens]
        if first + len(tokens) > self.max_context:
            raise Refused("positions %d .. %d pass the context of %d" % (first, first + len(tokens) - 1, self.max_context))
        if sample is not None:
            raise Refused("this program's head is the greedy one or the logits: no sampler on the card yet")
        out = None
        if not rows or self.SOCKETS > 1 or len(tokens) < self.ROWS_FROM:
            for i, t in enumerate(tokens):
                out = self.step(t, first + i, greedy=greedy, head=i == len(tokens) - 1)
            return out
        card = self.place(self.SEAT)[0]
        with self._card_locks[card]:
            return self._prompt_rows(tokens, first, greedy, card)

    def _prompt_rows(self, tokens, first, greedy, card):
        """`prefill`'s chunks on card `card`, which the caller holds."""
        out = None
        prev2 = self._at("prev2")
        back = [int(t) for t in np.frombuffer(self.m.read(prev2, 8), dtype="<i4")]
        for a in range(0, len(tokens), self.CHUNK):
            part = tokens[a:a + self.CHUNK]
            seen = back + part
            self.m.write(self.AT_OF[card]["ngrR"], _half(self.ng.rows(self.ng.chunk_ids(seen, 2, len(part)))))
            back = seen[-2:]
            self.m.write(prev2, np.array(back, dtype="<i4").tobytes())
            last = a + len(part) == len(tokens)
            on = int(self.PROMPT_CARD and len(part) >= self.CARD_FROM)
            given = " ".join("(sys__node_array__set toks %d %d)" % (r, t) for r, t in enumerate(part))
            out = self.m.run_in(card, self._inline("(begin %s (prompt_rows %d %d %d) (ends %d))"
                                                   % (given, first + a, len(part), on, self._end(greedy, last))),
                                view=self.view_of(self.SEAT))[1]
        return out

    def _run(self, text, what):
        """`generate`'s program (▶ generate.py) on the seat `on_seat` chose, on its card's block in its view — the CPU
        worker takes each layer's experts — its answer."""
        card = self.place(self.SEAT)[0]
        with self._card_locks[card]:
            return self.m.run_in(card, self._inline(text), view=self.view_of(self.SEAT))[1]

    def logits(self, seat=None):
        """The logits a head left on seat `seat`'s card — the seat `on_seat` chose's by default."""
        return np.frombuffer(self.m.read(self._at("logits_h", seat), self.VOCAB * 2), dtype="<f2").astype(np.float32)
