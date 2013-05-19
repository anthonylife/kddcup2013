/* Copyright (c) Team project for 2013 KDDCup2013 Track 1.
   Author: anthonylife

   We adopt a new method to prepare data for feature extraction, that is
   creating tables by input SQL sentence to PostgreSQL and then output
   the constructed tables to disk.
*/


/* create author_year_count table */
select authorid, year, count(*) as count into authoryearcnt
    from paperauthor pa
    LEFT OUTER JOIN paper p on p.id = pa.paperid
    group by authorid, year
    order by authorid, year;

/* create autor_paper_year */
select authorid, paperid, year into authorpaperyear
    from paperauthor pa
    LEFT OUTER JOIN paper p on p.id = pa.paperid
    order by authorid, paperid, year;

/* export tables to files */
\copy (select authorid "authorid", year "year", count "count" from authoryearcnt) to '../../features/authoryearcnt.csv' with csv header;
\copy (select authorid "authorid", paperid "paperid", year "year" from authorpaperyear) to '../../features/authorpaperyear.csv' with csv header;

/* clean */
drop table authoryearcnt;
drop table authorpaperyear;
