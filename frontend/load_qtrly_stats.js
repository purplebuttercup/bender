// Define dimensions of the plot
var margin = {top: 120, right: 60, bottom: 60, left: 180};
var height = 900;
var width = 860;

// Define the transition duration
var transDur = 500;

// Set up a global variable for the names of the stats reported here
// (in hopes of making it easier to keep line colors consistent
var reportStats = [];

var stats;


// Load in the CRD quarterly stats table
d3.csv("test.csv", function(crd) {

    // Format the variables as neeeded
    crd.forEach(function(d) {
        d.stat_year = +d.stat_year;
        d.stat_qtr = +d.stat_qtr;
        d.datestring = " Team" + d.stat_qtr;
        d.qtr_result = +d.qtr_result;
        //console.log(d)
    });

    var qtrly = crd.filter(function(d) {
        return (d.stat_name == "Maintenance" ||
            d.stat_name == "Engine" ||
            d.stat_name == "Engine Electrical System" ||
            d.stat_name == "Fuel Preparation and Regulation" ||
            d.stat_name == "Fuel Suply System" ||
            d.stat_name == "Cooling System" ||
            d.stat_name == "Exhaust System" ||
            d.stat_name == "Clutch" ||
            d.stat_name == "Engine and Transmission Suspension" ||
            d.stat_name == "Manual Transmission" ||
            d.stat_name == "Automatic Transmission" ||
            d.stat_name == "Gearshift Mechanism" ||
            d.stat_name == "Propeller Shaft" ||
            d.stat_name == "Transfer Box" ||
            d.stat_name == "Double-clutch gearbox" ||
            d.stat_name == "Front Axle" ||
            d.stat_name == "Steering and Wheel Alignment" ||
            d.stat_name == "Rear Axle" ||
            d.stat_name == "Brakes" ||
            d.stat_name == "Pedals" ||
            d.stat_name == "Wheels and Tyres" ||
            d.stat_name == "Integrated Suspension Systems" ||
            d.stat_name == "Body" ||
            d.stat_name == "Body Equipment" ||
            d.stat_name == "Seats" ||
            d.stat_name == "Slide/Tilt Sunroof+Convertible Top" ||
            d.stat_name == "Slide/General Electrical System" ||
            d.stat_name == "Instruments" ||
            d.stat_name == "Lights" ||
            d.stat_name == "Heating and Air Conditioning" ||
            d.stat_name == "Audio, Navigation, Information Systems" ||
            d.stat_name == "Distance Systems, Cruise Control, Remote Control" ||
            d.stat_name == "Electrical Drives" ||
            d.stat_name == "Parts+Accessories (Engine+Chassis)" ||
            d.stat_name == "Parts+Accessories (Body)" ||
            d.stat_name == "Communication Systems" ||
            d.stat_name == "Body Cavity Sealing / Undercoating" ||
            d.stat_name == "Paintwork"
        )

            ; });


    // Assign the data outside of the function for later use
    stats = qtrly;


    // Load the initial stats table
    makeMultiTable(stats);


    // Make a vector for all of the stats, so that plot attributes can be
    // kept consistent - probably a better way to do this.
    d3.selectAll("tbody tr")
        .each(function(d) { reportStats.push(d.key); });


    // Setup the line plot
    setupPlot(height, width, margin);


});
