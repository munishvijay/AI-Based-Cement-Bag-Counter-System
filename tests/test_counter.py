import unittest
from app import CrossingCounter

class CrossingCounterTests(unittest.TestCase):
    def test_first_crossing_counted_only_once(self):
        c=CrossingCounter(0.5)
        self.assertFalse(c.update(8,20,100))
        self.assertTrue(c.update(8,80,100))
        self.assertFalse(c.update(8,20,100))
        self.assertEqual(c.counted_ids,{8})

    def test_tracks_are_independent(self):
        c=CrossingCounter(0.5)
        c.update(1,20,100); c.update(2,80,100)
        self.assertTrue(c.update(1,80,100))
        self.assertFalse(c.update(2,90,100))
        self.assertTrue(c.update(2,20,100))

    def test_no_crossing_means_no_count(self):
        c=CrossingCounter()
        self.assertFalse(c.update(3,10,100))
        self.assertFalse(c.update(3,12,100))
        self.assertEqual(c.counted_ids,set())

    def test_invalid_line_position(self):
        with self.assertRaises(ValueError): CrossingCounter(1.2)

if __name__=="__main__": unittest.main()
