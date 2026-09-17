"""Benchmark primitives: small Ising Hamiltonians whose terms are measured through ancillas.

A primitive is a set of *terms*.  Each term is one ancilla plus the data qubits whose
``Z...Z`` product it realises: during an LR-QAOA layer the data parities are copied onto the
ancilla, the ancilla is rotated by ``2 gamma``, measured mid-circuit, and the byproduct
``Z...Z`` is undone by feed-forward.  That is a stabiliser measurement with a phase kick, so
the same machinery covers every family we want to benchmark:

* ``Chain``       - 1D Ising chain ``d0-a0-d1-a1-...-d(n-1)``; the *triplet* is ``n = 2``;
* ``Direct``      - the same Ising chain without ancillas, every bond one native two-qubit
  rotation: the reference the measurement-based version is judged against;
* ``CodePatch``   - the checks of a QEC code (``qecbench.codes``: surface, colour, qLDPC or any
  structure you load), each check a term of 2, 4, 5, 6... data qubits, in either kind.

Everything downstream (layout search, packing, circuit builders, analysis) only talks to the
``Primitive`` interface, so adding a family is adding a subclass.

Instances live on **physical** qubits of a device.  The Hamiltonian is expressed on *data
indices* ``0..n-1`` (the order of ``data_qubits``), which is also the left-to-right order of
every bitstring the analysis sees.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cached_property


@dataclass(frozen=True)
class Term:
    """One ``weight * Z...Z`` term: the data qubits it acts on, and the ancilla that measures it.

    ``ancilla = None`` means the term is applied **directly** to its data qubits (one native
    two-qubit rotation per bond, or a CNOT ladder for more than two) instead of through a
    mid-circuit measurement.
    """

    ancilla: int | None
    data: tuple[int, ...]
    weight: float = 1.0


class Primitive:
    """An instance of a benchmark primitive placed on physical qubits.

    Three labels describe an instance, and they organise the result files:

    * ``structure`` - the *problem*: ``chain`` today, ``surface_code`` / ``color_code`` later.
      A ``Direct`` instance solves the same 1D chain as a ``Chain``, so both share it.
    * ``kind`` - the *implementation*: ``mcm`` for the measurement gadget, ``direct`` for the
      ancilla-free reference.
    * ``family`` - the class itself, used as the token in the file name.

    Results are stored as ``<backend>/<structure>/<kind>/<file>.json``.
    """

    family: str = "primitive"
    structure: str = "primitive"
    kind: str = "mcm"

    def __init__(self, qubits):
        self.qubits = tuple(int(q) for q in qubits)
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"{self.family}: repeated qubit in {self.qubits}")

    # -- structure, provided by each family --------------------------------------------
    @cached_property
    def terms(self) -> tuple[Term, ...]:
        raise NotImplementedError

    @cached_property
    def data_qubits(self) -> tuple[int, ...]:
        raise NotImplementedError

    # -- derived ------------------------------------------------------------------------
    @cached_property
    def ancillas(self) -> tuple[int, ...]:
        """The measured qubits - empty for a family whose terms are applied directly."""
        return tuple(t.ancilla for t in self.terms if t.ancilla is not None)

    @property
    def n_data(self) -> int:
        return len(self.data_qubits)

    @cached_property
    def couplers(self) -> tuple[tuple[int, int], ...]:
        """Every coupler the circuit uses, as sorted pairs.

        With an ancilla those are the (data, ancilla) couplers of the gadget; without one,
        the bonds between consecutive data qubits of the term.
        """
        pairs = set()
        for term in self.terms:
            if term.ancilla is None:
                pairs |= {tuple(sorted(p)) for p in itertools.pairwise(term.data)}
            else:
                pairs |= {tuple(sorted((d, term.ancilla))) for d in term.data}
        return tuple(sorted(pairs))

    @cached_property
    def hamiltonian(self) -> dict[tuple[int, ...], float]:
        """``{(i, j, ...): weight}`` on data indices: ``H = sum w Z_i Z_j ...``."""
        index = {q: i for i, q in enumerate(self.data_qubits)}
        return {tuple(index[d] for d in t.data): float(t.weight) for t in self.terms}

    @property
    def identity(self) -> tuple:
        """What distinguishes this instance from any other in a result file."""
        return (self.family, self.qubits)

    @property
    def tag(self) -> str:
        """File-name token identifying the instance, e.g. ``3_4_5``."""
        return "_".join(str(q) for q in self.qubits)

    def optimal_energy(self) -> float:
        """Ground-state energy of ``hamiltonian`` (brute force unless a family knows it)."""
        from .lrqaoa import energies

        return float(energies(self.hamiltonian, self.n_data).min())

    def max_energy(self) -> float:
        """Highest energy - the reference point ``r = 0`` of the approximation ratio."""
        from .lrqaoa import energies

        return float(energies(self.hamiltonian, self.n_data).max())

    # -- (de)serialisation ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"family": self.family, "qubits": list(self.qubits)}

    # An instance addresses like the plain qubit tuple it replaces, so ``t[1]`` is still the
    # ancilla of a triplet and ``Direct(chain)`` reads the chain's path straight off it.
    def __iter__(self):
        return iter(self.qubits)

    def __getitem__(self, index):
        return self.qubits[index]

    def __len__(self):
        return len(self.qubits)

    def __eq__(self, other):
        return type(self) is type(other) and self.qubits == other.qubits

    def __hash__(self):
        return hash((self.family, self.qubits))

    def __lt__(self, other):
        return (self.family, self.qubits) < (other.family, other.qubits)

    def __repr__(self):
        return f"{type(self).__name__}{self.qubits}"


class Chain(Primitive):
    """1D Ising chain on the physical path ``d0 - a0 - d1 - a1 - ... - d(n-1)``.

    ``H = sum_i Z_di Z_d(i+1)``, one ancilla per bond.  ``n_data = 2`` is the triplet
    ``(d1, a, d2)`` of ``benchmarking_ibm_auto.ipynb``; longer chains are the
    ``qaoa_1d_mcm`` circuits of ``benchmarking_ibm.ipynb``.

    The orientation is canonicalised (the lexicographically smaller of the path and its
    reverse), so a triplet is stored as ``(d1, a, d2)`` with ``d1 < d2`` - the same tag as
    the legacy ``*_tri_d1_a_d2_*`` result files.
    """

    family = "chain"
    structure = "chain"

    def __init__(self, qubits):
        qubits = tuple(int(q) for q in qubits)
        if len(qubits) < 3 or len(qubits) % 2 == 0:
            raise ValueError(f"a chain has 2n-1 >= 3 qubits (data, ancilla, ..., data); got {qubits}")
        super().__init__(min(qubits, qubits[::-1]))

    @classmethod
    def from_roles(cls, data, ancillas):
        """Interleave explicit data and ancilla lists into a chain."""
        if len(ancillas) != len(data) - 1:
            raise ValueError("a chain needs exactly one ancilla per bond")
        path = [data[0]]
        for a, d in zip(ancillas, data[1:]):
            path += [a, d]
        return cls(path)

    @cached_property
    def data_qubits(self):
        return self.qubits[0::2]

    @cached_property
    def terms(self):
        return tuple(Term(self.qubits[i], (self.qubits[i - 1], self.qubits[i + 1]))
                     for i in range(1, len(self.qubits), 2))

    def optimal_energy(self):
        return -float(self.n_data - 1)   # antiferromagnetic order, frustration free

    def max_energy(self):
        return float(self.n_data - 1)


class Direct(Primitive):
    """The **direct** implementation of the same Ising chain: no ancilla, no measurement.

    The qubits are adjacent and all carry data: every bond's ``exp(-i gamma Z Z)`` is one
    native two-qubit rotation, so there is no ancilla, no mid-circuit measurement and no
    feed-forward anywhere.

    ``H = sum_i Z_i Z_{i+1}`` over ``len(qubits)`` data qubits, so a ``Direct`` instance of
    ``n`` qubits and a ``Chain`` of ``n`` data qubits run **the same logical circuit** and
    share the noiseless reference; ``Direct.from_chain`` builds exactly that counterpart.
    The difference between their measured ``r`` is what the measure-and-correct cycle costs.
    """

    family = "direct"
    structure = "chain"        # the same 1D chain, implemented without an ancilla
    kind = "direct"

    def __init__(self, qubits):
        qubits = tuple(int(q) for q in qubits)
        if len(qubits) < 2:
            raise ValueError(f"a direct chain needs at least two coupled qubits; got {qubits}")
        super().__init__(min(qubits, qubits[::-1]))

    @classmethod
    def from_chain(cls, chain, start=0):
        """The counterpart of ``chain``: **the same Hamiltonian**, no ancilla.

        Same number of data qubits, hence the same ``H``, the same noiseless reference and a
        directly comparable ``r``.  The physical qubits cannot be the chain's own data
        qubits - those are two hops apart, with the ancilla between them, so no native
        two-qubit gate reaches across.  The instance therefore takes ``n_data`` *adjacent*
        qubits of the same path (from ``start``), keeping it on the same patch of the chip:
        a triplet ``(d1, a, d2)`` gives the coupled pair ``(d1, a)``.

        Per layer and per bond this costs one two-qubit rotation against the chain's
        rotation plus a mid-circuit measurement and its feed-forward - which is the
        comparison the benchmark is about.
        """
        qubits = chain.qubits[start:start + chain.n_data]
        if len(qubits) < chain.n_data:
            raise ValueError(f"{chain} is too short to take {chain.n_data} qubits from {start}")
        return cls(qubits)

    @cached_property
    def data_qubits(self):
        return self.qubits

    @cached_property
    def terms(self):
        return tuple(Term(None, (self.qubits[i], self.qubits[i + 1]))
                     for i in range(len(self.qubits) - 1))

    def optimal_energy(self):
        return -float(self.n_data - 1)

    def max_energy(self):
        return float(self.n_data - 1)


class CodePatch(Primitive):
    """The checks of a QEC code (``codes.CodeStructure``) placed on qubits, in one kind.

    ``kind="mcm"`` gives every check an ancilla: its gadget is one round of syndrome extraction.
    ``kind="direct"`` applies each check as a CNOT ladder on its data qubits, no ancilla. Both
    solve the same Hamiltonian and share the noiseless reference.

    ``data_qubits[i]`` is the qubit of the structure's data qubit ``i``, and ``ancillas[c]`` the
    ancilla of check ``c``. The defaults are **logical** labels - data ``0..n-1``, ancillas
    ``n..n+m-1`` - for devices without a fixed layout (Quantinuum). On a fixed-coupling device
    they are physical qubits (see ``layout.surface_code_placements``).
    """

    family = "code"

    def __init__(self, code, data_qubits=None, ancillas=None, kind="mcm"):
        if kind not in ("mcm", "direct"):
            raise ValueError(f"unknown kind {kind!r}")
        self.code, self.kind = code, kind
        n, m = code.n_data, code.n_checks
        data = tuple(range(n)) if data_qubits is None else tuple(int(q) for q in data_qubits)
        if len(data) != n:
            raise ValueError(f"{code.name}: {len(data)} data qubits for n_data = {n}")
        if kind == "mcm":
            anc = tuple(range(n, n + m)) if ancillas is None else tuple(int(q) for q in ancillas)
            if len(anc) != m:
                raise ValueError(f"{code.name}: {len(anc)} ancillas for {m} checks")
        else:
            if ancillas:
                raise ValueError("a direct code patch has no ancillas")
            anc = ()
        self._data, self._anc = data, anc
        super().__init__(data + anc)

    @property
    def structure(self) -> str:
        return self.code.family

    @cached_property
    def data_qubits(self):
        return self._data

    @cached_property
    def terms(self):
        return tuple(Term(self._anc[c] if self.kind == "mcm" else None,
                          tuple(self._data[q] for q in check), w)
                     for c, (check, w) in enumerate(zip(self.code.checks, self.code.weights)))

    @property
    def logical(self) -> bool:
        """Whether the labels are the default logical ones."""
        n, m = self.code.n_data, self.code.n_checks
        return (self._data == tuple(range(n))
                and self._anc == (tuple(range(n, n + m)) if self.kind == "mcm" else ()))

    @property
    def identity(self):
        return (self.family, self.code.name, self.kind, self.qubits)

    @property
    def tag(self) -> str:
        return f"{self.code.name}_{self.kind}" + ("" if self.logical else f"_{min(self.qubits)}")

    def optimal_energy(self):
        return self.code.optimal_energy()

    def max_energy(self):
        return self.code.max_energy()

    def to_dict(self):
        return {"family": self.family, "kind": self.kind, "code": self.code.to_dict(),
                "data_qubits": list(self._data), "ancillas": list(self._anc), "qubits": list(self.qubits)}

    @classmethod
    def from_dict(cls, spec):
        from .codes import CodeStructure

        return cls(CodeStructure.from_dict(spec["code"]), spec["data_qubits"], spec["ancillas"] or None,
                   spec["kind"])

    def __eq__(self, other):
        return isinstance(other, CodePatch) and self.identity == other.identity

    def __hash__(self):
        return hash(self.identity)

    def __lt__(self, other):
        if isinstance(other, CodePatch):
            return self.identity < other.identity
        return (self.family, self.qubits) < (other.family, other.qubits)

    def __repr__(self):
        where = "logical" if self.logical else f"qubits {min(self.qubits)}..{max(self.qubits)}"
        return f"CodePatch({self.code.name}, {self.kind}, {where})"


FAMILIES = {"chain": Chain, "direct": Direct, "code": CodePatch}


def from_dict(spec) -> Primitive:
    """Inverse of ``Primitive.to_dict``; a bare qubit list is read as a chain."""
    if isinstance(spec, (list, tuple)):
        return Chain(spec)
    if spec["family"] == "code":
        return CodePatch.from_dict(spec)
    return FAMILIES[spec["family"]](spec["qubits"])
