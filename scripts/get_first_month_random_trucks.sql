-- This scripts filters the data_input database to obtain a sample from the first month and 10% random plates (9 plates)
-- This line was almost entirely suggested by GitHub Copilot - God damn it's good (it also auto completed this comment)
SELECT * FROM data_input WHERE ((timestamp BETWEEN '2021-10-14' AND '2021-11-14') AND (plate IN (SELECT plate FROM data_input WHERE (timestamp BETWEEN '2021-10-14' AND '2021-11-14') ORDER BY RAND() LIMIT 9)));


-- Now we do it in two lines
-- The first line saves the random list of plates to a temporary table
-- The second line filters the data_input database to obtain a sample from the first month and 10% random plates (9 plates)
-- This is the same as the previous script, but it's more readable
CREATE TEMPORARY TABLE random_plates AS SELECT plate FROM data_input WHERE (timestamp BETWEEN '2021-10-14' AND '2021-11-14') ORDER BY RAND() LIMIT 9;
SELECT * FROM data_input WHERE ((timestamp BETWEEN '2021-10-14' AND '2021-11-14') AND (plate IN (SELECT plate FROM random_plates)));