#!/usr/bin/python3
# -*- coding: utf-8 -*-
from llama_index.schema import NodeRelationship, RelatedNodeInfo, TextNode


def main():
    node1 = TextNode(text="<text_chunk>", id_="<node_id>")
    node2 = TextNode(text="<text_chunk>", id_="<node_id>")

    # set relationships
    node1.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=node2.node_id, metadata={"key": "val"})
    node2.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=node1.node_id, metadata={"key": "val"})

    print(node1)
    return


if __name__ == '__main__':
    main()
