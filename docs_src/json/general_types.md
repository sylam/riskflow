# General JSON Datatypes

All data types used in RiskFlow are the standard JSON types (string, float, integers) except
for the following:

- ModelParams. An object with the following fields:
    - modeldefaults: a JSON object with a data type (as a string) as the fieldname and its default 
as the value (also a string).
    - modelfilters: a JSON object with a data type (again as a string) as the fieldname and an array 
of fieldname, value pairs followed by a particular override (as a string).
- Curve. An object with two fields:
    - meta: usually set to [] if the curve is 1 dimensional. 
    - data: an array with floats or integers. Can be a list of pairs, triples or quads.
- Percent. Float value interpreted as being entered in percentage points.
- Basis. Float value interpreted as being entered in basis points.
- DateOffset. A string describing the time period in pandas DateOffset format:
    - days: D
    - weeks: W
    - months: M
    - years: Y 
    - e.g. 3 months 2 days would be 3M2D
- Offset. Identical to DateOffset but used to define Grids (defined below)
- Grid: An array of DateOffset and Offset types used to define a time grid for simulation purposes.
    - Usually, when specifying a time grid, a set of DateOffsets relative to the base_date are required
    - should you wish to specify recurring offsets relative to these DateOffsets, then use an Offset
    - e.g. instead of specifying a grid that's 7D, 14D, 21D ... (i.e. every 7 days) you can simply 
  encode it as ``` 
  {
    ".Grid": [
      {
        ".DateOffset": "7D",
        ".Offset": "1W"
      }
    ]
  }``` 
    - More complex time grids can be created by combining successive DateOffset and Offset types.
- Timestamp: String value interpreted as a date in YYYY-MM-DD date format
- Deal: A JSON object that may include one or more of the previous types used to specify a 
financial instrument.

Note that all objects must be preceded with a '.'(period) as the fieldname in the JSON. 