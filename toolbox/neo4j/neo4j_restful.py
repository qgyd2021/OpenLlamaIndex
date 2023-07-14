#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import logging
from typing import List, Union
import time

import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger('neo4j')


class Neo4jRestful(object):
    """
    http://localhost:7474/browser/

    """
    headers = {
        'Content-Type': 'application/json'
    }

    def __init__(self,
                 host: str = '127.0.0.1',
                 port: int = 7474,
                 database: str = 'default',
                 username: str = 'neo4j',
                 password: str = 'neo4j',
                 ):
        self.url = 'http://{host}:{port}/'.format(host=host, port=port)

        self.auth = HTTPBasicAuth(
            username=username,
            password=password,
        )

        resp = requests.get(url=self.url, headers=self.headers)
        if resp.status_code != 200:
            message = 'init request failed, status code: {}, text: {}'.format(resp.status_code, resp.text)
            logger.error(message)
            raise AssertionError(message)

        js = resp.json()
        transaction: str = js['transaction']
        self.transaction_url = transaction.format(databaseName=database)

        logger.info('neo4j restful initialized')
        logger.info('url: {}'.format(self.url))
        logger.info('transaction_url: {}'.format(self.transaction_url))

    def _retry_to_ensure_cmd(self, statements: Union[str, List[str]], do_commit: bool = False):
        retry_count = 0
        while True:
            try:
                result = self._cmd(statements, do_commit)
                break
            except requests.exceptions.ConnectionError as e:
                logging.error(e)

                retry_count += 1
                time.sleep(retry_count * 1.0)
                continue

        return result

    def _cmd(self, statements: Union[str, List[str]], do_commit: bool = False):
        if isinstance(statements, str):
            statements = [{'statement': statements, 'includeStats': True}]
        else:
            statements = [{'statement': statement, 'includeStats': True} for statement in statements]

        data = {
            "statements": statements
        }
        data = json.dumps(data)
        logger.debug(data)

        resp = requests.post(url=self.transaction_url, headers=self.headers, auth=self.auth, data=data)

        if resp.status_code not in (200, 201):
            logger.error('transaction failed, status_code: {}, text: {}'.format(resp.status_code, resp.text))
            js = None
            commit_url = None
        else:
            js = resp.json()
            commit_url = js['commit']

        if do_commit and commit_url is not None:
            self.commit(commit_url)
        return js

    def cmd(self, statements: Union[str, List[str]], do_commit: bool = False, retry_to_ensure_success: bool = False):
        if retry_to_ensure_success:
            result = self._retry_to_ensure_cmd(statements, do_commit)
        else:
            result = self._cmd(statements, do_commit)
        return result

    def commit(self, commit_url: str):
        resp = requests.post(url=commit_url, headers=self.headers, auth=self.auth)
        if resp.status_code != 200:
            logger.error('commit failed, status_code: {}, text: {}'.format(resp.status_code, resp.text))
            result = None
        else:
            js = resp.json()
            result = js
        return result


def demo1():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    neo4j_restful = Neo4jRestful(
        database='neo4j',
        username='neo4j',
        password='Glory@2021!'
    )

    # 创建
    # statement = 'CREATE (x:Human:Chinese:Programmer:田兴)'
    # result = neo4j_restful.cmd(statements=statement, do_commit=True)
    # print(result)
    # statement = 'CREATE (x:Human:Chinese:Programmer:宋军)'
    # result = neo4j_restful.cmd(statements=statement, do_commit=True)
    # print(result)

    # 删除
    # statement = 'MATCH (x:Human:Chinese:Programmer:田兴) DELETE x'
    # result = neo4j_restful.cmd(statements=statement, do_commit=True)
    # print(result)

    # 删除所有节点和关系
    statement = 'MATCH (n) DETACH DELETE (n)'

    result = neo4j_restful.cmd(statements=statement, do_commit=True)
    print(result)

    # 查找
    # statement = 'MATCH (x:田兴) RETURN COUNT(x)'
    # statement = 'MATCH (x:Programmer) RETURN COUNT(x)'
    # result = neo4j_restful.cmd(statements=statement, do_commit=True)
    # print(result)

    # 没有时创建
    # statement = 'MERGE (x:曾日强) RETURN x'
    # result = neo4j_restful.cmd(statements=statement, do_commit=True)
    # print(result)

    return


if __name__ == '__main__':
    demo1()
