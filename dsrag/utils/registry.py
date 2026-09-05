"""
One serializable-component registry, shared by every pluggable family.

``Embedding``, ``Reranker``, ``LLM``, ``VectorDB``, ``ChunkDB``, ``FileSystem``
and ``VLM`` each carried their own copy of the same twenty lines: a class-level
``subclasses`` dict, an ``__init_subclass__`` that fills it, a ``to_dict`` that
names the subclass, and a ``from_dict`` that looks it up. Seven copies is seven
chances to drift, and they had:

* Six popped ``subclass_name`` out of the caller's dict, mutating a config the
  caller still held — ``KnowledgeBase._load`` passes a live slice of the loaded
  metadata straight in, so loading a KB stripped the component names out of it.
* Six reported a missing ``subclass_name`` as ``Unknown subclass: None``, which
  reads like a bad name rather than an absent key.
* None of them said which names *were* registered, so a typo gave no direction.

The shared base takes the best of each: registration by class name, a config the
caller keeps intact, and a failure that names the key or lists the alternatives.

Each family gets its OWN registry. A direct subclass of
``SerializableComponent`` heads a family and opens a registry for it, so
``Reranker.from_dict`` can never build an ``Embedding`` — and two families may
each hold a class called ``PostgresDB`` without colliding.
"""

from typing import Any, ClassVar


class SerializableComponent:
    """A pluggable component that can be named in a config and rebuilt from it."""

    #: Registered subclasses of this family, keyed by class name. Each family
    #: head gets a fresh dict; members share their head's.
    subclasses: ClassVar[dict[str, type["SerializableComponent"]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if SerializableComponent in cls.__bases__:
            cls.subclasses = {}
        else:
            cls.subclasses[cls.__name__] = cls

    def to_dict(self) -> dict[str, Any]:
        """
        The config that rebuilds this instance.

        The base records only which subclass to build. A subclass that takes
        constructor arguments overrides this and adds them, conventionally as
        ``{**super().to_dict(), "argument": self.argument}``.
        """
        return {"subclass_name": self.__class__.__name__}

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "SerializableComponent":
        """
        Build the subclass the config names, passing the rest as keyword arguments.

        The config is read, never modified — the caller keeps whatever they
        passed in.

        Raises:
            ValueError: if ``subclass_name`` is absent, or names nothing this
                family has registered.
        """
        subclass_name = config.get("subclass_name")
        if not subclass_name:
            raise ValueError(
                f"{cls.__name__} config must include 'subclass_name'"
            )
        subclass = cls.subclasses.get(subclass_name)
        if subclass is None:
            known = ", ".join(sorted(cls.subclasses)) or "none registered"
            raise ValueError(
                f"Unknown subclass: {subclass_name}. "
                f"Registered {cls.__name__} subclasses are: {known}"
            )
        arguments = {key: value for key, value in config.items() if key != "subclass_name"}
        return subclass(**arguments)
