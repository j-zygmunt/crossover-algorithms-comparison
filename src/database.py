import sqlite3

def get_db_connection(db_path: str) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(db_path)
        print("Database created and Successfully Connected to SQLite")
        return connection
    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)


def prepare_db(connection: sqlite3.Connection) -> None:
    with open('db\scripts\prepare_db.sql', 'r') as script_file:
        script = script_file.read()
    execute_script(connection, script)


def clear_db(connection: sqlite3.Connection) -> None:
    with open('db\scripts\clear_db.sql', 'r') as script_file:
        script = script_file.read()
    execute_script(connection, script)


def execute_script(connection: sqlite3.Connection, script: str) -> None:
    cursor = connection.cursor()
    cursor.executescript(script)
    cursor.close()


def insert_experiment_data(connection: sqlite3.Connection, data: dict)-> None:
    columns = ', '.join(data.keys())
    names = ", ".join("?" * len(data.keys()))
    values = list(data.values())

    cursor = connection.cursor()
    cursor.execute(f"INSERT INTO experiments ({columns}) VALUES ({names})", values)
    cursor.close()
    connection.commit()
