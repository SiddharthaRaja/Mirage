"""A tiny, synthetic, Spider-shaped dataset that ships with the repo.

Eight invented databases, ~100 questions, every row of data made up. It
exists so that the whole pipeline — loading, schema rendering, execution,
suite building, rewards, the audit, evaluation and the report — runs and is
tested on any laptop and in CI without downloading anything.

It deliberately reproduces the property of Spider that makes this project
interesting: **the tables are tiny**. Spider's databases often have a handful
of rows per table, and on a handful of rows a wrong query frequently returns
the right answer by coincidence. The fixture's data is small for the same
reason, and a few questions are built so that a near-miss query matches on
the original database (a dropped ORDER BY whose first row happens to be the
maximum, a filter whose correct answer is empty). Those are what the audit
should catch.

Nothing measured on this fixture is a result about Spider or about any model.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from mirage.data.schema import tables_json_entry

# Each database: DDL, rows per table, and (question, gold SQL) pairs.
DATABASES: dict[str, dict] = {
    # ------------------------------------------------------------------ train
    "concert_hall": {
        "split": "train",
        "ddl": [
            "CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, country TEXT, age INTEGER, is_male INTEGER)",
            "CREATE TABLE concert (concert_id INTEGER PRIMARY KEY, concert_name TEXT, venue TEXT, year INTEGER)",
            "CREATE TABLE singer_in_concert (concert_id INTEGER, singer_id INTEGER, PRIMARY KEY (concert_id, singer_id), "
            "FOREIGN KEY (concert_id) REFERENCES concert(concert_id), FOREIGN KEY (singer_id) REFERENCES singer(singer_id))",
        ],
        "rows": {
            # singer 1 is both first and oldest: dropping ORDER BY still "works"
            "singer": [
                (1, "Mara Quell", "France", 52, 0),
                (2, "Tomas Ries", "Netherlands", 32, 1),
                (3, "Ilse Varga", "France", 41, 0),
                (4, "Joon Park", "South Korea", 29, 1),
                (5, "Ana Duarte", "Portugal", 25, 0),
                (6, "Leon Abara", "Nigeria", 43, 1),
            ],
            "concert": [
                (1, "Summer Lights", "Harbour Stage", 2014),
                (2, "Night Bloom", "Old Mill", 2015),
                (3, "Echo Park", "Harbour Stage", 2015),
                (4, "Winter Glass", "City Dome", 2016),
            ],
            "singer_in_concert": [(1, 1), (1, 2), (2, 3), (2, 2), (3, 6), (4, 4), (4, 1)],
        },
        "questions": [
            ("How many singers are there?", "SELECT count(*) FROM singer"),
            ("List the names of singers older than 40.", "SELECT name FROM singer WHERE age > 40"),
            ("What is the average age of singers from France?", "SELECT avg(age) FROM singer WHERE country = 'France'"),
            ("What is the name of the oldest singer?", "SELECT name FROM singer ORDER BY age DESC LIMIT 1"),
            ("How many concerts were held in each year?", "SELECT year, count(*) FROM concert GROUP BY year"),
            ("List all distinct countries that singers come from.", "SELECT DISTINCT country FROM singer"),
            (
                "What are the names of singers who performed in the concert 'Summer Lights'?",
                "SELECT T1.name FROM singer AS T1 JOIN singer_in_concert AS T2 ON T1.singer_id = T2.singer_id "
                "JOIN concert AS T3 ON T2.concert_id = T3.concert_id WHERE T3.concert_name = 'Summer Lights'",
            ),
            (
                "For each concert, show its name and the number of singers who performed.",
                "SELECT T2.concert_name, count(*) FROM singer_in_concert AS T1 JOIN concert AS T2 "
                "ON T1.concert_id = T2.concert_id GROUP BY T1.concert_id",
            ),
            (
                "Which venue hosted the most concerts?",
                "SELECT venue FROM concert GROUP BY venue ORDER BY count(*) DESC LIMIT 1",
            ),
            (
                "What are the names of singers who did not perform in any concert?",
                "SELECT name FROM singer WHERE singer_id NOT IN (SELECT singer_id FROM singer_in_concert)",
            ),
            ("Which countries have more than one singer?", "SELECT country FROM singer GROUP BY country HAVING count(*) > 1"),
            ("What are the minimum and maximum ages of male singers?", "SELECT min(age), max(age) FROM singer WHERE is_male = 1"),
            ("List the names of singers from Japan.", "SELECT name FROM singer WHERE country = 'Japan'"),
            (
                "List concert names held after 2014, ordered by year.",
                "SELECT concert_name FROM concert WHERE year > 2014 ORDER BY year",
            ),
        ],
    },
    "library": {
        "split": "train",
        "ddl": [
            "CREATE TABLE author (author_id INTEGER PRIMARY KEY, name TEXT, birth_year INTEGER, nationality TEXT)",
            "CREATE TABLE book (book_id INTEGER PRIMARY KEY, title TEXT, author_id INTEGER, genre TEXT, pages INTEGER, "
            "published_year INTEGER, FOREIGN KEY (author_id) REFERENCES author(author_id))",
            "CREATE TABLE loan (loan_id INTEGER PRIMARY KEY, book_id INTEGER, member_name TEXT, loan_year INTEGER, "
            "returned INTEGER, FOREIGN KEY (book_id) REFERENCES book(book_id))",
        ],
        "rows": {
            "author": [
                (1, "Odile Marsh", 1948, "British"),
                (2, "Kenji Arata", 1971, "Japanese"),
                (3, "Paula Svensk", 1985, "Swedish"),
                (4, "Rui Coelho", 1962, "Portuguese"),
            ],
            "book": [
                (1, "The Salt Road", 1, "Fiction", 412, 1999),
                (2, "Paper Lanterns", 2, "Fiction", 288, 2008),
                (3, "Counting Rivers", 3, "Science", 350, 2016),
                (4, "Quiet Engines", 2, "Science", 198, 2012),
                (5, "A Tide of Names", 1, "History", 520, 2004),
                (6, "Small Hours", 4, "Fiction", 240, 2019),
            ],
            "loan": [
                (1, 1, "Dev", 2021, 1),
                (2, 2, "Ines", 2021, 1),
                (3, 1, "Ines", 2022, 0),
                (4, 4, "Marta", 2022, 1),
                (5, 6, "Dev", 2023, 0),
            ],
        },
        "questions": [
            ("How many books are in the library?", "SELECT count(*) FROM book"),
            ("List the titles of all fiction books.", "SELECT title FROM book WHERE genre = 'Fiction'"),
            ("What is the title of the longest book?", "SELECT title FROM book ORDER BY pages DESC LIMIT 1"),
            ("How many books are there in each genre?", "SELECT genre, count(*) FROM book GROUP BY genre"),
            (
                "What are the titles of books written by Odile Marsh?",
                "SELECT T1.title FROM book AS T1 JOIN author AS T2 ON T1.author_id = T2.author_id WHERE T2.name = 'Odile Marsh'",
            ),
            ("What is the average number of pages of science books?", "SELECT avg(pages) FROM book WHERE genre = 'Science'"),
            ("How many loans have not been returned?", "SELECT count(*) FROM loan WHERE returned = 0"),
            (
                "Which authors have written more than one book? Give their names.",
                "SELECT T2.name FROM book AS T1 JOIN author AS T2 ON T1.author_id = T2.author_id GROUP BY T1.author_id HAVING count(*) > 1",
            ),
            (
                "List the titles of books that have never been loaned.",
                "SELECT title FROM book WHERE book_id NOT IN (SELECT book_id FROM loan)",
            ),
            ("List the names of authors born before 1970, from oldest to youngest.", "SELECT name FROM author WHERE birth_year < 1970 ORDER BY birth_year"),
            ("Which member names have borrowed books?", "SELECT DISTINCT member_name FROM loan"),
            ("What are the titles of books published after 2020?", "SELECT title FROM book WHERE published_year > 2020"),
        ],
    },
    "airline": {
        "split": "train",
        "ddl": [
            "CREATE TABLE airport (airport_code TEXT PRIMARY KEY, city TEXT, country TEXT)",
            "CREATE TABLE aircraft (aircraft_id INTEGER PRIMARY KEY, model TEXT, capacity INTEGER)",
            "CREATE TABLE flight (flight_no TEXT PRIMARY KEY, origin TEXT, destination TEXT, aircraft_id INTEGER, distance INTEGER, "
            "price REAL, FOREIGN KEY (origin) REFERENCES airport(airport_code), FOREIGN KEY (destination) REFERENCES airport(airport_code), "
            "FOREIGN KEY (aircraft_id) REFERENCES aircraft(aircraft_id))",
        ],
        "rows": {
            "airport": [
                ("HYD", "Hyderabad", "India"),
                ("BLR", "Bengaluru", "India"),
                ("DXB", "Dubai", "UAE"),
                ("SIN", "Singapore", "Singapore"),
                ("LHR", "London", "UK"),
            ],
            "aircraft": [(1, "A320", 180), (2, "B737", 162), (3, "A350", 325)],
            "flight": [
                ("MR101", "HYD", "BLR", 1, 500, 3200.0),
                ("MR102", "BLR", "HYD", 1, 500, 3100.0),
                ("MR201", "HYD", "DXB", 2, 2600, 14500.0),
                ("MR301", "BLR", "SIN", 3, 3200, 18900.5),
                ("MR401", "DXB", "LHR", 3, 5500, 31000.0),
                ("MR402", "HYD", "SIN", 2, 2900, 15800.0),
            ],
        },
        "questions": [
            ("How many flights are there?", "SELECT count(*) FROM flight"),
            ("List the flight numbers of flights departing from HYD.", "SELECT flight_no FROM flight WHERE origin = 'HYD'"),
            ("What is the average price of all flights?", "SELECT avg(price) FROM flight"),
            ("Which aircraft model has the largest capacity?", "SELECT model FROM aircraft ORDER BY capacity DESC LIMIT 1"),
            (
                "List the cities of airports in India.",
                "SELECT city FROM airport WHERE country = 'India'",
            ),
            (
                "Show the destination city of flight MR301.",
                "SELECT T2.city FROM flight AS T1 JOIN airport AS T2 ON T1.destination = T2.airport_code WHERE T1.flight_no = 'MR301'",
            ),
            ("How many flights does each aircraft model operate?", "SELECT T2.model, count(*) FROM flight AS T1 JOIN aircraft AS T2 ON T1.aircraft_id = T2.aircraft_id GROUP BY T1.aircraft_id"),
            ("What is the longest flight distance?", "SELECT max(distance) FROM flight"),
            ("List flight numbers of flights cheaper than 5000.", "SELECT flight_no FROM flight WHERE price < 5000"),
            (
                "Which airports have no departing flights? Give the airport codes.",
                "SELECT airport_code FROM airport WHERE airport_code NOT IN (SELECT origin FROM flight)",
            ),
            ("List all flights ordered by distance from shortest to longest. Show the flight numbers.", "SELECT flight_no FROM flight ORDER BY distance ASC"),
            ("How many distinct countries have airports?", "SELECT count(DISTINCT country) FROM airport"),
        ],
    },
    "school_club": {
        "split": "train",
        "ddl": [
            "CREATE TABLE student (student_id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT, age INTEGER, major TEXT, city TEXT)",
            "CREATE TABLE club (club_id INTEGER PRIMARY KEY, club_name TEXT, club_location TEXT)",
            "CREATE TABLE member_of_club (student_id INTEGER, club_id INTEGER, position TEXT, PRIMARY KEY (student_id, club_id), "
            "FOREIGN KEY (student_id) REFERENCES student(student_id), FOREIGN KEY (club_id) REFERENCES club(club_id))",
        ],
        "rows": {
            "student": [
                (1, "Asha", "Rao", 19, "Biology", "Pune"),
                (2, "Ben", "Cole", 21, "Physics", "Delhi"),
                (3, "Chen", "Wu", 20, "Biology", "Pune"),
                (4, "Dina", "Haddad", 22, "Mechanical", "Chennai"),
                (5, "Eli", "Stone", 18, "Physics", "Delhi"),
                (6, "Farah", "Noor", 20, "Mechanical", "Hyderabad"),
            ],
            "club": [(1, "Robotics", "Workshop A"), (2, "Chess", "Library"), (3, "Debate", "Hall 2")],
            "member_of_club": [(1, 1, "President"), (4, 1, "Member"), (6, 1, "Member"), (2, 2, "President"), (3, 2, "Member"), (5, 3, "Member")],
        },
        "questions": [
            ("How many students are there?", "SELECT count(*) FROM student"),
            ("List the first names of students majoring in Biology.", "SELECT first_name FROM student WHERE major = 'Biology'"),
            ("What is the average age of students?", "SELECT avg(age) FROM student"),
            ("How many students are in each major?", "SELECT major, count(*) FROM student GROUP BY major"),
            (
                "List the first names of members of the Robotics club.",
                "SELECT T1.first_name FROM student AS T1 JOIN member_of_club AS T2 ON T1.student_id = T2.student_id "
                "JOIN club AS T3 ON T2.club_id = T3.club_id WHERE T3.club_name = 'Robotics'",
            ),
            ("Which club has the most members?", "SELECT T2.club_name FROM member_of_club AS T1 JOIN club AS T2 ON T1.club_id = T2.club_id GROUP BY T1.club_id ORDER BY count(*) DESC LIMIT 1"),
            ("Who is the youngest student? Give the first and last name.", "SELECT first_name, last_name FROM student ORDER BY age LIMIT 1"),
            ("List the distinct cities students come from.", "SELECT DISTINCT city FROM student"),
            ("List the last names of students older than 20 who live in Delhi.", "SELECT last_name FROM student WHERE age > 20 AND city = 'Delhi'"),
            ("Which students are not members of any club? Give first names.", "SELECT first_name FROM student WHERE student_id NOT IN (SELECT student_id FROM member_of_club)"),
            ("List the names of club presidents.", "SELECT T1.first_name FROM student AS T1 JOIN member_of_club AS T2 ON T1.student_id = T2.student_id WHERE T2.position = 'President'"),
            ("How many students are older than 25?", "SELECT count(*) FROM student WHERE age > 25"),
        ],
    },
    "car_shop": {
        "split": "train",
        "ddl": [
            "CREATE TABLE maker (maker_id INTEGER PRIMARY KEY, name TEXT, country TEXT)",
            "CREATE TABLE model (model_id INTEGER PRIMARY KEY, maker_id INTEGER, model_name TEXT, horsepower INTEGER, price REAL, "
            "year INTEGER, FOREIGN KEY (maker_id) REFERENCES maker(maker_id))",
        ],
        "rows": {
            "maker": [(1, "Avanti", "Italy"), (2, "Kogane", "Japan"), (3, "Nordvik", "Sweden"), (4, "Ruhr", "Germany")],
            "model": [
                (1, 1, "Strada", 210, 41000.0, 2019),
                (2, 1, "Vela", 340, 88000.0, 2021),
                (3, 2, "Mizu", 120, 19000.0, 2018),
                (4, 2, "Kaze", 150, 23500.0, 2021),
                (5, 3, "Fjord", 190, 36000.0, 2020),
                (6, 4, "Werk", 260, 52000.0, 2022),
                (7, 2, "Tsuki", 95, 15500.0, 2017),
            ],
        },
        "questions": [
            ("How many car models are there?", "SELECT count(*) FROM model"),
            ("List the names of models with more than 200 horsepower.", "SELECT model_name FROM model WHERE horsepower > 200"),
            ("What is the most expensive model?", "SELECT model_name FROM model ORDER BY price DESC LIMIT 1"),
            ("How many models does each maker produce? Give maker name and count.", "SELECT T2.name, count(*) FROM model AS T1 JOIN maker AS T2 ON T1.maker_id = T2.maker_id GROUP BY T1.maker_id"),
            ("What is the average price of models made in 2021?", "SELECT avg(price) FROM model WHERE year = 2021"),
            ("List the model names made by Japanese makers.", "SELECT T1.model_name FROM model AS T1 JOIN maker AS T2 ON T1.maker_id = T2.maker_id WHERE T2.country = 'Japan'"),
            ("Which maker has the most models?", "SELECT T2.name FROM model AS T1 JOIN maker AS T2 ON T1.maker_id = T2.maker_id GROUP BY T1.maker_id ORDER BY count(*) DESC LIMIT 1"),
            ("What are the lowest and highest horsepower values?", "SELECT min(horsepower), max(horsepower) FROM model"),
            ("List model names ordered by year, newest first.", "SELECT model_name FROM model ORDER BY year DESC"),
            ("Which makers are from Germany or Sweden?", "SELECT name FROM maker WHERE country = 'Germany' OR country = 'Sweden'"),
            ("List models cheaper than 10000.", "SELECT model_name FROM model WHERE price < 10000"),
        ],
    },
    "football_league": {
        "split": "train",
        "ddl": [
            "CREATE TABLE team (team_id INTEGER PRIMARY KEY, name TEXT, city TEXT, founded INTEGER)",
            "CREATE TABLE match (match_id INTEGER PRIMARY KEY, home_team INTEGER, away_team INTEGER, home_goals INTEGER, away_goals INTEGER, "
            "season INTEGER, FOREIGN KEY (home_team) REFERENCES team(team_id), FOREIGN KEY (away_team) REFERENCES team(team_id))",
        ],
        "rows": {
            "team": [(1, "Harbour Rovers", "Kochi", 1931), (2, "Plateau United", "Pune", 1958), (3, "Red Fort FC", "Delhi", 1990), (4, "Deccan Stars", "Hyderabad", 2004)],
            "match": [
                (1, 1, 2, 2, 1, 2023),
                (2, 3, 4, 0, 0, 2023),
                (3, 2, 3, 3, 2, 2023),
                (4, 4, 1, 1, 4, 2024),
                (5, 1, 3, 2, 2, 2024),
                (6, 2, 4, 1, 0, 2024),
            ],
        },
        "questions": [
            ("How many teams are in the league?", "SELECT count(*) FROM team"),
            ("Which team was founded first?", "SELECT name FROM team ORDER BY founded LIMIT 1"),
            ("How many matches were played in each season?", "SELECT season, count(*) FROM match GROUP BY season"),
            ("How many matches ended in a draw?", "SELECT count(*) FROM match WHERE home_goals = away_goals"),
            ("What is the total number of goals scored by home teams?", "SELECT sum(home_goals) FROM match"),
            ("List the names of teams founded after 1950.", "SELECT name FROM team WHERE founded > 1950"),
            ("Which teams played a home match in 2024? Give the team names.", "SELECT DISTINCT T2.name FROM match AS T1 JOIN team AS T2 ON T1.home_team = T2.team_id WHERE T1.season = 2024"),
            ("In which match were the most goals scored? Give the match id.", "SELECT match_id FROM match ORDER BY home_goals + away_goals DESC LIMIT 1"),
            ("List the cities of all teams in alphabetical order.", "SELECT city FROM team ORDER BY city"),
            ("Which teams never played a home match? Give names.", "SELECT name FROM team WHERE team_id NOT IN (SELECT home_team FROM match)"),
            ("What is the average number of away goals per match?", "SELECT avg(away_goals) FROM match"),
        ],
    },
    # -------------------------------------------------------------------- dev
    "museum_visits": {
        "split": "dev",
        "ddl": [
            "CREATE TABLE museum (museum_id INTEGER PRIMARY KEY, name TEXT, open_year INTEGER, num_staff INTEGER)",
            "CREATE TABLE visitor (visitor_id INTEGER PRIMARY KEY, name TEXT, membership_level INTEGER, age INTEGER)",
            "CREATE TABLE visit (museum_id INTEGER, visitor_id INTEGER, num_tickets INTEGER, total_spent REAL, PRIMARY KEY (museum_id, visitor_id), "
            "FOREIGN KEY (museum_id) REFERENCES museum(museum_id), FOREIGN KEY (visitor_id) REFERENCES visitor(visitor_id))",
        ],
        "rows": {
            "museum": [(1, "Glass House", 1998, 40), (2, "River Archive", 2008, 15), (3, "Stone Gallery", 2012, 22), (4, "Map Room", 2015, 9)],
            "visitor": [(1, "Ira", 3, 34), (2, "Joel", 1, 28), (3, "Kavya", 4, 51), (4, "Lior", 2, 19), (5, "Mina", 4, 45)],
            "visit": [(1, 1, 2, 40.0), (1, 3, 5, 125.5), (2, 3, 1, 15.0), (3, 2, 3, 60.0), (3, 5, 4, 98.0), (2, 4, 1, 12.0)],
        },
        "questions": [
            ("How many museums are there?", "SELECT count(*) FROM museum"),
            ("List the names of museums opened after 2010.", "SELECT name FROM museum WHERE open_year > 2010"),
            ("What is the average number of staff in museums?", "SELECT avg(num_staff) FROM museum"),
            ("Which museum has the most staff?", "SELECT name FROM museum ORDER BY num_staff DESC LIMIT 1"),
            ("List visitor names with membership level higher than 3, ordered by age.", "SELECT name FROM visitor WHERE membership_level > 3 ORDER BY age"),
            ("What is the total amount spent across all visits?", "SELECT sum(total_spent) FROM visit"),
            ("Which visitors visited the Glass House museum? Give names.", "SELECT T2.name FROM visit AS T1 JOIN visitor AS T2 ON T1.visitor_id = T2.visitor_id JOIN museum AS T3 ON T1.museum_id = T3.museum_id WHERE T3.name = 'Glass House'"),
            ("Which museums have never been visited? Give names.", "SELECT name FROM museum WHERE museum_id NOT IN (SELECT museum_id FROM visit)"),
            ("How many visits did each museum get? Show museum name and count.", "SELECT T2.name, count(*) FROM visit AS T1 JOIN museum AS T2 ON T1.museum_id = T2.museum_id GROUP BY T1.museum_id"),
            ("What is the name of the visitor who spent the most in a single visit?", "SELECT T2.name FROM visit AS T1 JOIN visitor AS T2 ON T1.visitor_id = T2.visitor_id ORDER BY T1.total_spent DESC LIMIT 1"),
            ("List the names of visitors younger than 18.", "SELECT name FROM visitor WHERE age < 18"),
            ("How many visitors have membership level 4?", "SELECT count(*) FROM visitor WHERE membership_level = 4"),
        ],
    },
    "restaurant_chain": {
        "split": "dev",
        "ddl": [
            "CREATE TABLE restaurant (rest_id INTEGER PRIMARY KEY, name TEXT, city TEXT, rating REAL)",
            "CREATE TABLE dish (dish_id INTEGER PRIMARY KEY, rest_id INTEGER, dish_name TEXT, price REAL, is_vegetarian INTEGER, "
            "FOREIGN KEY (rest_id) REFERENCES restaurant(rest_id))",
        ],
        "rows": {
            "restaurant": [(1, "Saffron Table", "Hyderabad", 4.6), (2, "Coastline", "Kochi", 4.1), (3, "Iron Wok", "Pune", 3.8), (4, "Leaf & Lime", "Hyderabad", 4.4)],
            "dish": [
                (1, 1, "Haleem", 320.0, 0),
                (2, 1, "Mirchi Salan", 210.0, 1),
                (3, 2, "Fish Moilee", 450.0, 0),
                (4, 3, "Hakka Noodles", 260.0, 1),
                (5, 4, "Green Bowl", 300.0, 1),
                (6, 4, "Paneer Tikka", 280.0, 1),
                (7, 2, "Appam", 150.0, 1),
            ],
        },
        "questions": [
            ("How many restaurants are there?", "SELECT count(*) FROM restaurant"),
            ("List the names of restaurants in Hyderabad.", "SELECT name FROM restaurant WHERE city = 'Hyderabad'"),
            ("Which restaurant has the highest rating?", "SELECT name FROM restaurant ORDER BY rating DESC LIMIT 1"),
            ("How many vegetarian dishes are there?", "SELECT count(*) FROM dish WHERE is_vegetarian = 1"),
            ("What is the average dish price?", "SELECT avg(price) FROM dish"),
            ("List dish names served at Coastline.", "SELECT T1.dish_name FROM dish AS T1 JOIN restaurant AS T2 ON T1.rest_id = T2.rest_id WHERE T2.name = 'Coastline'"),
            ("How many dishes does each restaurant serve? Give name and count.", "SELECT T2.name, count(*) FROM dish AS T1 JOIN restaurant AS T2 ON T1.rest_id = T2.rest_id GROUP BY T1.rest_id"),
            ("What is the cheapest dish?", "SELECT dish_name FROM dish ORDER BY price LIMIT 1"),
            ("Which cities have more than one restaurant?", "SELECT city FROM restaurant GROUP BY city HAVING count(*) > 1"),
            ("List restaurants with no vegetarian dishes.", "SELECT name FROM restaurant WHERE rest_id NOT IN (SELECT rest_id FROM dish WHERE is_vegetarian = 1)"),
            ("List dish names costing more than 1000.", "SELECT dish_name FROM dish WHERE price > 1000"),
        ],
    },
}


def build_mini_spider(out_dir: str | Path) -> Path:
    """Write the fixture to disk in Spider's layout. Idempotent."""
    root = Path(out_dir)
    db_root = root / "database"
    db_root.mkdir(parents=True, exist_ok=True)
    tables, train, dev = [], [], []
    for db_id, spec in DATABASES.items():
        d = db_root / db_id
        d.mkdir(exist_ok=True)
        path = d / f"{db_id}.sqlite"
        if path.exists():
            path.unlink()
        con = sqlite3.connect(path)
        try:
            for stmt in spec["ddl"]:
                con.execute(stmt)
            for table, rows in spec["rows"].items():
                if rows:
                    ph = ",".join("?" * len(rows[0]))
                    con.executemany(f"INSERT INTO {table} VALUES ({ph})", rows)
            con.commit()
        finally:
            con.close()
        tables.append(tables_json_entry(db_id, path))
        target = train if spec["split"] == "train" else dev
        for q, sql in spec["questions"]:
            target.append({"db_id": db_id, "question": q, "query": sql})
    (root / "tables.json").write_text(json.dumps(tables, indent=1), encoding="utf-8")
    (root / "train_spider.json").write_text(json.dumps(train, indent=1), encoding="utf-8")
    (root / "dev.json").write_text(json.dumps(dev, indent=1), encoding="utf-8")
    (root / "README.txt").write_text(
        "Synthetic Spider-shaped fixture generated by mirage. All data invented.\n"
        "Not Spider. Nothing measured on it is a result about Spider or any model.\n",
        encoding="utf-8",
    )
    return root
