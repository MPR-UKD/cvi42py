import argparse
import pickle
import numpy as np
from xml.dom import minidom
from typing import List, Dict


def keep_element_nodes(nodes: minidom.NodeList) -> List[minidom.Node]:
    """Filter out and keep only element nodes from a list of nodes.

    Args:
        nodes (minidom.NodeList): A list of nodes to be filtered.

    Returns:
        List[minidom.Node]: A list of element nodes.
    """
    return [node for node in nodes if node.nodeType == node.ELEMENT_NODE]


def parse_contours(node: minidom.Node) -> Dict[str, np.ndarray]:
    """Parse a Contours object to extract contour points.

    Each Contours object may contain several contours. This function parses
    the contour name, then the points and pixel size.

    Args:
        node (minidom.Node): The DOM node representing a Contours object.

    Returns:
        Dict[str, np.ndarray]: A dictionary mapping contour names to their points.
    """
    contours = {}
    for child in keep_element_nodes(node.childNodes):
        contour_name = child.getAttribute("Hash:key")
        sub = 1
        points = []

        for child2 in keep_element_nodes(child.childNodes):
            if child2.getAttribute("Hash:key") == "Points":
                for child3 in keep_element_nodes(child2.childNodes):
                    x = float(child3.getElementsByTagName("Point:x")[0].firstChild.data)
                    y = float(child3.getElementsByTagName("Point:y")[0].firstChild.data)
                    points.append([x, y])

            if child2.getAttribute("Hash:key") == "SubpixelResolution":
                sub = int(child2.firstChild.data)

        points = np.array(points) / sub
        contours[contour_name] = points

    return contours


def traverse_node(node: minidom.Node, uid_contours: Dict[str, Dict[str, np.ndarray]]):
    """Recursively traverse the nodes to extract contour information.

    Args:
        node (minidom.Node): The current DOM node.
        uid_contours (Dict[str, Dict[str, np.ndarray]]): A dictionary to store the extracted contours.
    """
    child = node.firstChild
    while child:
        if child.nodeType == child.ELEMENT_NODE and child.getAttribute("Hash:key") == "ImageStates":
            for child2 in keep_element_nodes(child.childNodes):
                uid = child2.getAttribute("Hash:key")
                for child3 in keep_element_nodes(child2.childNodes):
                    if child3.getAttribute("Hash:key") == "Contours":
                        contours = parse_contours(child3)
                        if contours:
                            uid_contours[uid] = contours
        traverse_node(child, uid_contours)
        child = child.nextSibling


def parse_file(xml_name: str, coord_file: str):
    """Parse a cvi42 XML file and save the extracted data to a pickle file.

    Args:
        xml_name (str): Path to the XML file to be parsed.
        coord_file (str): Path where the output pickle file will be saved.
    """
    dom = minidom.parse(xml_name)
    uid_contours = {}
    traverse_node(dom, uid_contours)

    with open(coord_file, "wb") as f:
        pickle.dump(uid_contours, f)


def main():
    """Command-line interface for the cvi42py tool."""
    parser = argparse.ArgumentParser(description="Process a cvi42 XML file and save contours to a pickle file.")
    parser.add_argument("xml_file", type=str, help="Path to the cvi42 XML file.")
    parser.add_argument("output_file", type=str, help="Path to the output pickle file.")
    args = parser.parse_args()

    parse_file(args.xml_file, args.output_file)


if __name__ == "__main__":
    main()
