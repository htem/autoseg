import sqlite3
import json


class Database:
    """
    Simple SQLite Database Wrapper for Storing and Retrieving Scores.

    This class provides a simple wrapper around an SQLite database for storing and retrieving scores.
    Each score entry is associated with a network, checkpoint, threshold, and a dictionary of scores.

    Args:
        db_name (str):
            The name of the SQLite database file.
        table_name (str):
            The name of the table within the database (default is 'scores_table').

    Attributes:
        conn (sqlite3.Connection):
            The SQLite database connection.
        cursor (sqlite3.Cursor):
            The SQLite database cursor.
        table_name (str):
            The name of the table within the database.

    Methods:
        add_score(network, checkpoint, threshold, scores_dict):
            Add a score entry to the database.

        get_scores(networks=None, checkpoints=None, thresholds=None):
            Retrieve scores from the database based on specified conditions.

    """

    def __init__(self, db_name, table_name="scores_table"):
        self.conn = sqlite3.connect(f"{db_name}.db", check_same_thread=False)
        self.table_name = table_name

        self.cursor = self.conn.cursor()
        # check if table exists
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [k[0] for k in self.cursor.fetchall()]
        if self.table_name not in tables:
            self.cursor.execute(
                f"CREATE TABLE {self.table_name} (network text, checkpoint int, threshold real, scores text)"
            )
            self.conn.commit()

    def add_score(self, network, checkpoint, threshold, scores_dict):
        """
        Add a score entry to the database.

        Args:
            network (str):
                The name of the network.
            checkpoint (int):
                The checkpoint number.
            threshold (float):
                The threshold value.
            scores_dict (dict):
                A dictionary containing scores.
        """
        assert type(network) is str
        assert type(checkpoint) is int
        assert type(threshold) is float
        assert type(scores_dict) is dict
        scores_str = json.dumps(scores_dict)
        self.cursor.execute(
            f"INSERT INTO {self.table_name} VALUES ('{network}', {checkpoint}, {threshold}, '{scores_str}')"
        )
        self.conn.commit()

    def get_scores(self, networks=None, checkpoints=None, thresholds=None):
        """
        Retrieve scores from the database based on specified conditions.

        Args:
            networks (str, list):
                The name or list of names of networks to filter on.
            checkpoints (int, list):
                The checkpoint number or list of checkpoint numbers to filter on.
            thresholds (float, list):
                The threshold value or list of threshold values to filter on.

        Returns:
            list: A list of tuples representing retrieved score entries.
        """
        assert type(networks) is str or networks is None or type(networks) is list
        assert (
            type(checkpoints) is int or checkpoints is None or type(checkpoints) is list
        )
        assert (
            type(thresholds) is float or thresholds is None or type(thresholds) is list
        )

        def to_csv_list(l):
            return ",".join([f"'{ll}'" for ll in l])

        conditioned = False

        def add_where(var, var_name, query):
            nonlocal conditioned
            ret = ""
            if var is not None:
                if conditioned:
                    ret += " and "
                else:
                    ret += " where "
                conditioned = True
                if type(var) is str:
                    ret += f"{var_name} = '{var}'"
                else:
                    ret += f"{var_name} in ({to_csv_list(var)})"
            return ret

        query = f"SELECT * FROM {self.table_name}"
        query += add_where(networks, "network", query)
        query += add_where(checkpoints, "checkpoint", query)
        query += add_where(thresholds, "threshold", query)

        ret = self.cursor.execute(query).fetchall()
        ret = [list(k) for k in ret]
        for item in ret:
            item[3] = json.loads(item[3])

        return ret
