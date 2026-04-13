-- Lua filter for single-column layout
-- Ensures figures have proper width and tables don't overflow

function Figure(el)
  -- Render figure with full textwidth
  local rendered = pandoc.write(pandoc.Pandoc({el}), "latex")
  -- Ensure width is \textwidth
  rendered = rendered:gsub("\\linewidth", "\\textwidth")
  return pandoc.RawBlock("latex", rendered)
end
