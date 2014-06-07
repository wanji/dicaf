#!/usr/bin/env python
# coding: utf-8

#########################################################################
#########################################################################

"""
   File Name: level2hbase.py
      Author: Wan Ji
      E-mail: wanji@live.com
  Created on: Sat Jun  7 13:36:03 2014 CST
"""
DESCRIPTION = """
This program can transfer the data from LevelDB to HBase.
"""

import os
import sys
import argparse

import leveldb

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol


def perr(msg):
    """ Print error message.
    """

    sys.stderr.write("%s" % msg)
    sys.stderr.flush()


def pinfo(msg):
    """ Print information message.
    """

    sys.stdout.write("%s" % msg)
    sys.stdout.flush()


def runcmd(cmd):
    """ Run command.
    """

    perr("%s\n" % cmd)
    os.system(cmd)


def getargs():
    """ Parse program arguments.
    """

    parser = argparse.ArgumentParser(description=DESCRIPTION,
                                     formatter_class=
                                     argparse.RawTextHelpFormatter)
    parser.add_argument('leveldb', type=str,
                        help='path to the LevelDB database')
    parser.add_argument('table', type=str,
                        help='target table name in hbase')
    parser.add_argument('host', type=str, nargs='?', default="127.0.0.1",
                        help='IP address / Host name of hbase server')
    parser.add_argument('port', type=int, nargs='?', default=9090,
                        help='port number of  hbase server')
    parser.add_argument('pyhbase', type=str, nargs='?', default="gen-py",
                        help='python interface of hbase')

    return parser.parse_args()


def main(args):
    """ Main entry.
    """

    transport = TSocket.TSocket(args.host, args.port)
    transport = TTransport.TBufferedTransport(transport)
    protocol = TBinaryProtocol.TBinaryProtocol(transport)

    client = Hbase.Client(protocol)
    transport.open()

    contents = ColumnDescriptor(name='cf:', maxVersions=1)
    ldb = leveldb.LevelDB(args.leveldb)
    iter = ldb.RangeIter()
    try:
        client.createTable(args.table, [contents])
    except AlreadyExists as err:
        perr("ERROR: %s\n" % err.message)
        sys.exit(1)

    cnt = 0
    pinfo("Processed image:\n")
    pinfo("\r\t%d" % cnt)
    while True:
        try:
            item = iter.next()
        except StopIteration:
            break
        cnt += 1
        if cnt % 100 == 0:
            pinfo("\r\t%d" % cnt)
        client.mutateRow('cifar-train', item[0],
                         [Mutation(column="cf:data", value=item[1])], None)
    pinfo("\r\t%d\tDone!\n" % cnt)


if __name__ == '__main__':
    args = getargs()

    sys.path.append(args.pyhbase)
    from hbase import Hbase
    from hbase.ttypes import *

    main(args)
