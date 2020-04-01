import sys
import xml.etree.ElementTree as ET

'''
根据函数体源码生成的xml提取statement子树序列
'''


# 带双亲节点的树节点
class treeNode:
    def __init__(self, parent, ele):
        if parent != None:
            self.parent = parent
            self.ele = ele
        else:
            self.parent = parent
            self.ele = ele

# 根据根节点提取AST
def extractSTBaseRoot(root):

    statemnentTag = {"if","while","for","unit","switch"
                     }

    # 解析xml
    def parseXML(path):
        try:
            tree = ET.parse(path)
            # 获得根节点
            root = tree.getroot()
            return root
        except Exception as e:  # 捕获除与程序退出sys.exit()相关之外的所有 异常
            print("parse tbcnn.xmlProcess fail!")
            sys.exit()

    # 深度优先遍历树
    def traverse(node):
        print(node.tag)
        for childNode in node:
            traverse(childNode)

    # 根据深度优先遍历得到的列表，提取statement子树
    def extractStatement(tree):
        statementList = []
        for node in tree:
            if node.ele.tag in statemnentTag:
                statementList.append(node.ele)
                if node.parent != None:
                    node.parent.remove(node.ele)
        return statementList

    # 深度优先遍历树，树的节点为带双亲节点的结构
    def createTreeDeepFirst(root, list, parent):
        list.append(treeNode(parent, root))
        for node in root:
            createTreeDeepFirst(node, list, root)

    treeDeepFirstList = []
    createTreeDeepFirst(root, treeDeepFirstList, None)
    statementList = extractStatement(treeDeepFirstList)
    return statementList

# 根据树的根节点打印该函数的源码
# 方法： 如果当前标记的尾标记有内容，则先打印标记中的内容再打印尾标记
def traverse(root):
    if root.tail != None:
        if root.text != None:
            print(root.text, end="")
        for node in root:
            traverse(node)
        print(root.tail, end="")
    else:
        if root.text != None:
            print(root.text, end="")
        for node in root:
            traverse(node)

