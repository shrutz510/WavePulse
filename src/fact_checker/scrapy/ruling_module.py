from typing import Union, List, Dict


def create_new_ruling(data: Union[Dict, List]) -> Union[Dict, List]:
    """
    Replaces the 'ruling' key with 'ruling-unified' based on specific criteria in the given JSON data.

    Parameters:
        - data (Union[Dict, List]): The JSON data which can be either a dictionary or a list of dictionaries.

    Returns:
        - Union[Dict, List]: The modified JSON data with the 'ruling' key replaced by 'ruling-unified'.

    Example:
        >>> data = [
        >>>     {'title': 'Article 1', 'ruling': 'False'},
        >>>     {'title': 'Article 2', 'ruling': 'True'}
        >>> ]
        >>> create_new_ruling(data)
        [{'title': 'Article 1', 'ruling-unified': 'False'}, {'title': 'Article 2', 'ruling-unified': 'True'}]
    """

    if isinstance(data, dict):
        for key, value in list(data.items()):
            if key == "ruling" and isinstance(value, str):
                if value in ["True", "true", "Correct Attribution", "Recall", "Legit"]:
                    data["ruling-unified"] = "True"
                    data[
                        "ruling-description"
                    ] = "The statement is accurate and there’s nothing significant missing."
                elif value in ["False", "false", "Fake", "pants-fire", "Not True"]:
                    data["ruling-unified"] = "False"
                    data["ruling-description"] = "The statement is not accurate."
                elif value in ["Mostly True", "mostly-true", "Partly False"]:
                    data["ruling-unified"] = "Mostly True"
                    data[
                        "ruling-description"
                    ] = "The primary elements of a claim are true, but some of the ancillary details surrounding the claim may be inaccurate."
                elif value in [
                    "Mostly False",
                    "mostly-false",
                    "barely-true",
                    "Partially True",
                ]:
                    data["ruling-unified"] = "Mostly False"
                    data[
                        "ruling-description"
                    ] = "The primary elements of a claim are false, but some of the ancillary details surrounding the claim may be accurate."
                elif value in ["Mixed", "Mixture", "half-true", "Partially Verified"]:
                    data["ruling-unified"] = "Mixed"
                    data[
                        "ruling-description"
                    ] = "The claim has significant elements of both truth and falsity to it."
                elif value in ["Labeled Satire", "Originated as Satire"]:
                    data["ruling-unified"] = "Satire"
                    data[
                        "ruling-description"
                    ] = "The claim is derived from content described by its creator and/or the wider audience as satire."
                elif value in ["Miscaptioned", "Misattributed"]:
                    data["ruling-unified"] = "Miscaptioned"
                    data[
                        "ruling-description"
                    ] = "The quoted material, picture, or video has been incorrectly attributed to a person or wrongly captioned."
                elif value in ["Decontextualized"]:
                    data["ruling-unified"] = "Decontextualized"
                    data[
                        "ruling-description"
                    ] = "Taken out of the context in which it was originally intended."
                elif value == "only-analysis-no-label":
                    data["ruling-unified"] = "Only Analysis, No Label"
                    data[
                        "ruling-description"
                    ] = "Fact checked analyses of popular articles/ social media posts."
                elif value == "full-flop":
                    data["ruling-unified"] = "Full Flop"
                    data["ruling-description"] = "A complete change in position."
                elif value == "half-flip":
                    data["ruling-unified"] = "Half Flip"
                    data["ruling-description"] = "A partial change in position."
                elif value == "no-flip":
                    data["ruling-unified"] = "No Flip"
                    data["ruling-description"] = "No significant change in position."
                elif value in [
                    "Unverifiable",
                    "Unsubstantiated",
                    "Unproven",
                    "Unfounded",
                    "Undetermined",
                    "Legend",
                    "Research In Progress",
                    "True But Not Proven In Practice",
                    "Uncertain",
                    "Unknown",
                ]:
                    data["ruling-unified"] = "Unverified"
                    data[
                        "ruling-description"
                    ] = "Not found reliable or first-hand sources to satisfactorily confirm or deny the story."
                else:
                    data["ruling-unified"] = value
                    data["ruling-description"] = "Not Assigned"
            else:
                create_new_ruling(value)

    elif isinstance(data, list):
        for item in data:
            create_new_ruling(item)

    return data
