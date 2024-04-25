import re


def torchinfo2md(
    stats_str,
    row_seperetor="---",
    table_md_model="|---|---|---|",
    table_md_summary_header="|name|value|",
    table_md_summary="|---|---|",
):
    """change to md format of torchinfo result.

    Args:
        stats_str (str): torchinfo's string. get with str(torchinfo.summary(model,modelinput))
        row_seperetor (str, optional): use in each between of tables. Defaults to "---".
        table_md_model (str, optional): markdown table header for model table. Defaults to "|---|---|---|".
        table_md_summary_header (str, optional): markdown table header for summary table. Defaults to "|name|value|".
        table_md_summary (str, optional): markdown table header for summary table. Defaults to "|---|---|".

    Returns:
        str: md format string
    """

    formatted_lines = []
    stats_lines = stats_str.splitlines()

    # indexes for seperate with row separetor : "====="
    sep_indexes = [i for i, stats_line in enumerate(stats_lines) if "=" in stats_line]

    # make indexes sets of slices
    info_parts = list(zip(sep_indexes[:-1], sep_indexes[1:]))

    # model table's header
    model_header = stats_lines[slice(*info_parts[0])]

    formatted_lines.append("")
    formatted_lines.append(row_seperetor)
    header = model_header[1]

    separetor = "|"
    separetor_summary = ":"
    newline = "\n"

    # get indexes for insert seperetors
    matches = re.finditer(r"\s[\s]+[^\s]", header)
    spans = [m.span()[1] - 1 for m in matches]
    spans = [0] + spans + [None]
    sliceindexes = list(zip(spans[:-1], spans[1:]))

    # apply inserting separetors
    cols = [header[slice(*_s)] for _s in sliceindexes]
    col = separetor.join(cols)
    col = separetor + col + separetor

    formatted_lines.append(col)
    formatted_lines.append(table_md_model)

    # model table's body
    table_body_lines = stats_lines[slice(*info_parts[1])]
    # drop row separetor
    table_body_lines = table_body_lines[1:]

    # apply inserting separetors
    for table_body_line in table_body_lines:
        cols = [table_body_line[slice(*_s)] for _s in sliceindexes]
        col = separetor.join(cols)
        col = separetor + col + separetor
        formatted_lines.append(col)

    # summary table's body 1
    summary_lines = stats_lines[slice(*info_parts[2])]
    # drop row separetor
    summary_lines = summary_lines[1:]

    # insert new table header
    formatted_lines.append("")
    formatted_lines.append(row_seperetor)
    formatted_lines.append(table_md_summary_header)
    formatted_lines.append(table_md_summary)

    # insert seperator instead of ":"
    for summary_line in summary_lines:
        cols = summary_line.split(separetor_summary)
        col = separetor.join(cols)
        col = separetor + col + separetor
        formatted_lines.append(col)

    # summary table's body 2
    summary_lines = stats_lines[slice(*info_parts[3])]
    # drop row separetor
    summary_lines = summary_lines[1:]

    # insert new table header
    formatted_lines.append("")
    formatted_lines.append(row_seperetor)
    formatted_lines.append(table_md_summary_header)
    formatted_lines.append(table_md_summary)

    # insert seperator instead of ":"
    for summary_line in summary_lines:
        cols = summary_line.split(separetor_summary)
        col = separetor.join(cols)
        col = separetor + col + separetor
        formatted_lines.append(col)

    md_format = newline.join(formatted_lines)
    return md_format
