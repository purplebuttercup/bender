// This function creates a table with a row for each statistic in a flat data
// object and a column for each time period in the data object.

var makeMultiTable = function(stats) {

    // Set up the column names
    // One set for the year supercolumns


    // And one for the quarter columns
    var qtrCols = d3.keys(

        d3.nest()
            .key(function(d) {  return d.datestring; })
            .map(stats)
    );

    // Add an empty column for the statistic name
    qtrCols.unshift("");




    // Nest data within each statistic
    var aggstats = d3.nest()
        .key(function(d) { return d.stat_name; })
        .entries(stats)

    // Create the table
    var table = d3.select("#table").append("table").attr("class", "mdl-data-table mdl-js-data-table");
    var thead = table.append("thead");
    var tbody = table.append("tbody");


    // Append the quarter headers
    thead.append("tr")
        .selectAll("th")
        .data(qtrCols)
        .enter()
        .append("th")
        .attr("class", "mdl-data-table__cell--non-numeric")
        .text(function(column) { return column; })


    // Bind each statistic to a line of the table
    var rows = tbody.selectAll("tr")
        .data(aggstats)
        .enter()
        .append("tr")
        .attr("rowstat", function(d) { return d.key; })
        .attr("id", function(d) { return d.key; })
        .attr("chosen", false)
        .attr("onclick", function(d) {
            return "toggleStat('" + d.key + "')"; })


    // Add statistic names to each row
    var stat_cells = rows.append("td")
        .text(function(d) { return d.key; })
        .attr("class", "rowkey")
        .attr("class", "mdl-data-table__cell--non-numeric");


    // Fill in the cells.  The data -> d.values pulls the value arrays from
    // the data assigned above to each row.
    // The unshift crap bumps the data cells over one - otherwise, the first
    // result value falls under the statistic labels.
    var res_cells = rows.selectAll("td")
        .data(function(d) {
            var x = d.values;
            x.unshift({ qtr_result: ""} );
            return x; })
        .enter()
        .append("td")
        .text(function(d) { return d3.format(",d")(d.qtr_result); })




};