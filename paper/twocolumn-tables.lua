-- Lua filter: convert pandoc's longtable environments to table floats so that
-- tables don't split across pages unnecessarily. Large tables (>25 rows) are
-- left as longtable so they can break legitimately.

local function count_data_rows(tbl)
  local n = 0
  for _, body in ipairs(tbl.bodies or {}) do
    for _, row in ipairs(body.body or {}) do
      n = n + 1
    end
  end
  if tbl.foot and tbl.foot.rows then
    n = n + #tbl.foot.rows
  end
  return n
end

function Table(el)
  local rows = count_data_rows(el)
  local rendered = pandoc.write(pandoc.Pandoc({el}), "latex")

  if rows <= 25 then
    -- Extract caption if present
    local caption = ""
    rendered = rendered:gsub("\\caption{(.-)}\\tabularnewline\n", function(c)
      caption = c
      return ""
    end)

    -- Remove the longtable repeated-header block (between \endfirsthead and
    -- \endlastfoot) so single-page tables don't show a duplicate header.
    -- Pandoc structure:
    --   <first header>
    --   \endfirsthead
    --   <repeated header>
    --   \endhead
    --   \bottomrule\noalign{}
    --   \endlastfoot
    --   <data>
    -- We collapse everything between \endfirsthead and \endlastfoot (inclusive).
    rendered = rendered:gsub(
      "\\endfirsthead[%s%S]-\\endlastfoot\n",
      ""
    )

    -- Convert longtable to tabular
    rendered = rendered:gsub("\\begin{longtable}%[[^%]]*%]{([^}]*)}",
                             "\\begin{tabular}{%1}")
    rendered = rendered:gsub("\\begin{longtable}{([^}]*)}",
                             "\\begin{tabular}{%1}")
    rendered = rendered:gsub("\\end{longtable}",
                             "\\bottomrule\\noalign{}\n\\end{tabular}")

    -- Clean up longtable-specific commands that might remain
    rendered = rendered:gsub("\\endhead%s*", "")
    rendered = rendered:gsub("\\endfirsthead%s*", "")
    rendered = rendered:gsub("\\endfoot%s*", "")
    rendered = rendered:gsub("\\endlastfoot%s*", "")

    local result = "\\begin{table}[!htbp]\n\\centering\n"
    if caption ~= "" then
      result = result .. "\\caption{" .. caption .. "}\n"
    end
    result = result .. rendered .. "\n\\end{table}"
    return pandoc.RawBlock("latex", result)
  end
  return nil
end

function Figure(el)
  local rendered = pandoc.write(pandoc.Pandoc({el}), "latex")
  rendered = rendered:gsub("\\linewidth", "\\textwidth")
  return pandoc.RawBlock("latex", rendered)
end
