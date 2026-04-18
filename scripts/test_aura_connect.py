from __future__ import annotations

import logging
import os
import traceback

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from neo4j import GraphDatabase


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("neo4j").setLevel(logging.DEBUG)

    uri = os.getenv("NEO4J_URI", "").strip()
    user = os.getenv("NEO4J_USER", "").strip()
    password = os.getenv("NEO4J_PASSWORD", "")
    database = os.getenv("NEO4J_DATABASE", "").strip()

    print(f"URI: {uri or 'not set'}")
    print(f"USER: {user or 'not set'}")
    print(f"DATABASE: {database or 'not set'}")
    print(f"PASSWORD_SET: {bool(password)}")

    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("verify_connectivity(): OK")

        with driver.session(database=database or None) as session:
            record = session.run("RETURN 1 AS ok").single()
            print(f"RETURN 1 AS ok => {record['ok']}")
    except Exception as error:
        print("Connection failed:")
        print(f"{type(error).__name__}: {error}")
        print("Traceback:")
        print(traceback.format_exc())
    finally:
        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
