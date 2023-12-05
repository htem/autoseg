import unittest
import os
from autoseg.eval import Database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        # Create a test database
        self.db_name = "test_db"
        self.table_name = "test_scores_table"
        self.db = Database(self.db_name, self.table_name)

    def tearDown(self):
        # Remove the test database file after tests
        if os.path.exists(f"{self.db_name}.db"):
            os.remove(f"{self.db_name}.db")

    def test_add_score(self):
        # Test adding a score to the database
        network = "TestNetwork"
        checkpoint = 1
        threshold = 0.5
        scores_dict = {"metric": 0.8}
        self.db.add_score(network, checkpoint, threshold, scores_dict)

        # Verify that the score is added correctly
        result = self.db.get_scores(networks=network)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], network)
        self.assertEqual(result[0][1], checkpoint)
        self.assertEqual(result[0][2], threshold)
        self.assertEqual(result[0][3], scores_dict)

    def test_get_scores(self):
        # Test retrieving scores from the database
        network = "TestNetwork"
        checkpoint = 1
        threshold = 0.5
        scores_dict = {"metric": 0.8}
        self.db.add_score(network, checkpoint, threshold, scores_dict)

        # Verify that the retrieved score matches the added score
        result = self.db.get_scores(networks=network)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], network)
        self.assertEqual(result[0][1], checkpoint)
        self.assertEqual(result[0][2], threshold)
        self.assertEqual(result[0][3], scores_dict)

    def test_get_scores_multiple_conditions(self):
        # Test retrieving scores with multiple conditions
        network = "TestNetwork"
        checkpoint = 1
        threshold = 0.5
        scores_dict = {"metric": 0.8}
        self.db.add_score(network, checkpoint, threshold, scores_dict)

        # Add another score with different conditions
        network2 = "TestNetwork2"
        checkpoint2 = 2
        threshold2 = 0.7
        scores_dict2 = {"metric": 0.9}
        self.db.add_score(network2, checkpoint2, threshold2, scores_dict2)

        # Verify that the retrieved scores match the added scores with multiple conditions
        result = self.db.get_scores(networks=[network, network2], checkpoints=[checkpoint2])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], network2)
        self.assertEqual(result[0][1], checkpoint2)
        self.assertEqual(result[0][2], threshold2)
        self.assertEqual(result[0][3], scores_dict2)

