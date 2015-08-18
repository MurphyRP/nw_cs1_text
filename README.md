# nw_cs1_text

Script/App to consume corpus doc in JSON format, creates sparse vector matrix
with a 'column' for each term and row for each DCI and a separate list (array)
with the 'class'
assumes the following:

- ./cs1_terms.txt with \n 'terminated' file
- ./ccs1_DSI_JSON.txt with assumed JSON format
creates
- ./corpus with term weights per DCI - nothing is done with this
file, it is only for looking at specific outputs


### Version
1.0

### Tech

* JSON format - ('key_terms' not used in this script)

* Class can be any grouping - does not have to be binary
```
{"key_terms": {"noun_phrases": ["term"]}, "extracted": "text of DSE", "class": "Hillary"}
```
### Dependencies

You will need a few external libraries

```
from pandas import DataFrame
import numpy as np

```


