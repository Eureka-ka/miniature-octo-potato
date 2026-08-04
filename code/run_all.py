# -*- coding: utf-8 -*-
"""一键运行：问题1 -> 问题2(特征/统计) -> 问题3 -> 生成Word报告。"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    import problem1
    import problem2_features
    import problem2_stats
    import problem3
    import build_report
    print("\n########## 问题1 ##########")
    problem1.run()
    print("\n########## 问题2-特征 ##########")
    problem2_features.run()
    print("\n########## 问题2-统计 ##########")
    problem2_stats.run()
    print("\n########## 问题3 ##########")
    problem3.run()
    print("\n########## 生成报告 ##########")
    build_report.main()

if __name__ == "__main__":
    main()