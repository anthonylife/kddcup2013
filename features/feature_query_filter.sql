/* Copyright (c) Team project for 2013 KDDCup2013 Track 1.
   Author: anthonylife

   We adopt a new method to prepare data for feature extraction, that is
   creating tables by input SQL sentence to PostgreSQL and then output
   the constructed tables to disk.

   Different from the file "feature_query_unfilter.sql", this sql file
   contains the step of filtering users' documents to save cleaner documents.

   Note, currently, this file only support the database that deployed in
   our lab server with IP address, 166.111.68.43
*/


/* merge all (paperid, authorid) in train and valid files */
/*with temp as (
    select * from trainconfirmed union all 
    select * from traindeleted union all
    select * from validpaper
)
select * into trainvalid from temp order by authorid, paperid;*/


/* create filter_author_paper_year table */
with trainvalid_author_affi as (
    select distinct authorid, convert_from(affiliation, 'UTF8') as affiliation from trainvalid
),
paper_author_affi_year as (
    select paperid, authorid, convert_from(affiliation, 'UTF8') as affiliation, year
    from paperauthor pa
    LEFT OUTER JOIN paper p on p.id = pa.paperid
)
select t1.authorid, t2.paperid, t2.year into filter_author_paper from trainvalid_author_affi t1 join paper_author_affi_year t2 on t1.authorid=t2.authorid and (t1.affiliation=t2.affiliation or (t1.affiliation is null and t2.affiliation is null)); 


/* create author_year_count table */
select authorid, year, count(*) as count into author_year_cnt
    from filter_author_paper
    group by authorid, year
    order by authorid, year;


/* create autor_paper_year */
select authorid, paperid, year into author_paper_year
    from paperauthor pa
    LEFT OUTER JOIN paper p on p.id = pa.paperid
    order by authorid, paperid, year;


/* export tables to files */
\copy (select authorid "authorid", paperid "paperid" from filter_author_paper) to './filterauthorpaper.csv' with csv header;
\copy (select authorid "authorid", year "year", count "count" from author_year_cnt) to './authoryearcnt.csv' with csv header;
\copy (select authorid "authorid", paperid "paperid", year "year" from author_paper_year) to './authorpaperyear.csv' with csv header;

/* clean */
drop table filter_author_paper;
drop table author_year_cnt;
drop table author_paper_year;


