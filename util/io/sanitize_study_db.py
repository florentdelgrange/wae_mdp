import os
import sys

path = os.path.dirname(os.path.abspath("__file__"))
sys.path.insert(0, path + '/../../')

import sqlite3
from absl import app, flags


def main(argv):
    del argv
    params = FLAGS.flag_values_dict()

    con = sqlite3.connect(params['storage'])
    cur = con.cursor()
    cur.execute(f"SELECT MIN(VALUE) FROM trial_values WHERE value>{float(-1e10)}")
    min_val = cur.fetchall()[0][0]
    cur.execute(f"UPDATE trial_values SET value={min_val} WHERE value<{float(-1e10)}")
    con.commit()
    con.close()

    return 0


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    default_flags = FLAGS.flag_values_dict()

    flags.DEFINE_string(
        'storage',
        help='DB URL',
        default=None,
        required=True
    )

    FLAGS = flags.FLAGS

    app.run(main)
