import unittest
import OTSoft_file_reader
import MaxEnt


class TestGetInfo(unittest.TestCase):
    def test_basic_eq(self):
        output = OTSoft_file_reader.get_info("HayesPseudoKorean.txt")

        # The number of URs should equal to the number of tableaux.
        self.assertEqual(len(output[1]), len(output[3]))

        # The number of winners should equal or be larger than the number of URs.
        self.assertTrue(output[4] >= output[1])


if __name__ == "__main__":
    unittest.main()
