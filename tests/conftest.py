from __future__ import annotations

from collections import defaultdict

import pytest

from mirage.data.fixtures import build_mini_spider
from mirage.data.spider import Spider
from mirage.reward.suite import SuiteIndex, build_suite
from mirage.reward.verifier import Verifier


@pytest.fixture(scope="session")
def spider(tmp_path_factory) -> Spider:
    root = build_mini_spider(tmp_path_factory.mktemp("mini") / "spider")
    return Spider(root)


@pytest.fixture(scope="session")
def suites_dir(spider, tmp_path_factory):
    out = tmp_path_factory.mktemp("suites")
    by_db = defaultdict(list)
    for e in spider.load("train") + spider.load("dev"):
        by_db[e.db_id].append(e.gold_sql)
    for db, qs in by_db.items():
        build_suite(db, spider.db_path(db), qs, out, seed=1)
    return out


@pytest.fixture(scope="session")
def db_paths(spider):
    return {d: str(spider.db_path(d)) for d in spider.db_ids()}


@pytest.fixture()
def verifier(db_paths, suites_dir):
    return Verifier(db_paths, SuiteIndex(suites_dir))
