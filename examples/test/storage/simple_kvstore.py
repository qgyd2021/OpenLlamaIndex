#!/usr/bin/python3
# -*- coding: utf-8 -*-
from llama_index.storage.kvstore.simple_kvstore import SimpleKVStore


def main():
    """类似于 redis 里的 hash map.
    即 key-value 字典的 value 也是一个字典.
    用于表示 collection 和 key-value.
    """
    store = SimpleKVStore()

    store.put(key="key1", val={"val1_key1": "val1_val1"})
    store.put(key="key2", val={"val1_key2": "val1_val2"})

    val: dict = store.get(key="key1")
    print(val)

    all_val: dict = store.get_all()
    print(all_val)
    return


if __name__ == '__main__':
    main()
