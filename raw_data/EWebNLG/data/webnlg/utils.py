# -*- coding: utf-8 -*-
from __future__ import division, unicode_literals, print_function

import itertools
import re
import json

from enum import Enum
from typing import List, Tuple, Dict, Callable

import sys
import os.path

try:
    import spacy
except ImportError:
    os.system('pip install spacy')
    os.system('python -m spacy download en')
    import spacy

ALPHA = chr(2)  # Start of text
OMEGA = chr(3)  # End of text
SPLITABLES = {ALPHA, OMEGA, " ", ".", ",", ":", "-", "'", "(", ")", "?", "!",
              "&", ";", '"'}


class DataSetType(Enum):
    TEST = "test"
    TRAIN = "train"
    DEV = "dev"


class DataReader:

    def __init__(self, data: List[dict],
                 misspelling: Dict[str, str] = None,
                 rephrase: Tuple[Callable, Callable] = (None, None)):
        self.data = data
        self.misspelling = misspelling
        self.rephrase = rephrase

    def fix_spelling(self):
        if not self.misspelling:
            return self

        regex_splittable = "(\\" + "|\\".join(SPLITABLES) + ".)"

        for misspelling, fix in self.misspelling.items():
            source = regex_splittable + misspelling + regex_splittable
            target = "\1" + fix + "\2"

            self.data = [d.set_text(re.sub(source, target, d.text)) for d in
                         self.data]
        return self


class Cleaner():
    def __init__(self, verbose=False):
        self.fname_ends = [k[0] for k in self.filter_dic]
        # if verbose:
        #     keys = set(self.filter_dic.keys())
        #     with open('temp.txt') as f:
        #         data = [tuple(json.loads(line)) for line in f if line]
        #
        #     if set(data) != set(keys):
        #         import pdb;
        #         pdb.set_trace()
        #         set(keys) - set(data)


    def clean(self, filename):
        fname_end = '/'.join(filename.rsplit('/', 3)[1:])

        if fname_end not in self.fname_ends: return

        with open(filename, encoding="utf-8", errors='ignore') as f:
            lines = []
            content = f.readlines()
            for line_ix, line in enumerate(content):
                line = self.filter_line(fname_end, line_ix, line)
                if line: lines.append(line)
        if lines != content:
            # import pdb;
            # pdb.set_trace()
            fwrite(''.join(lines), filename)

    def filter_line(self, fname_end, line_ix, line):
        line = self.line_fix(line)

        text = line.strip()
        key = (fname_end, line_ix, text)
        if key in self.filter_dic:
            # fwrite(json.dumps(key) + '\n', 'temp.txt', mode='a')
            new_text = self.filter_dic[key]
            if not new_text: return False
            line = line.replace(text, new_text)

        return line

    @staticmethod
    def line_fix(line):
        line = line.replace(' (abbrv. Acta Palaeontol. Pol)',
                            ' (abbrv Acta Palaeontol Pol)')
        return line

    @property
    def filter_dic(self):
        return {
            ('dev/1triples/SportsTeam.xml', 185,
             '<sentence ID="1"/>'): '<sentence ID="1"><striple>A.E_Dimitra_Efxeinoupolis | fullname | &quot;A.E Dimitra Efxeinoupolis&quot;</striple></sentence>',
            ('dev/1triples/SportsTeam.xml', 197,
             '<sentence ID="1"/>'): '<sentence ID="1"><striple>A.E_Dimitra_Efxeinoupolis | fullname | &quot;A.E Dimitra Efxeinoupolis&quot;</striple></sentence>',
            ('dev/1triples/SportsTeam.xml', 208,
             '<sentence ID="1"/>'): '<sentence ID="1"><striple>A.E_Dimitra_Efxeinoupolis | fullname | &quot;A.E Dimitra Efxeinoupolis&quot;</striple></sentence>',
            ('dev/3triples/Airport.xml', 658, '<sentence ID="1"/>'): False, (
                'dev/3triples/Airport.xml', 659,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('dev/3triples/Building.xml', 104,
             '<template>AGENT-1 is located in BRIDGE-1, BRIDGE- in which PATIENT-2 are one of the ethnic groups and PATIENT-1 is the leader .</template>'): '<template>AGENT-1 is located in BRIDGE-1, a country in which PATIENT-2 are one of the ethnic groups and PATIENT-1 is the leader .</template>',
            ('dev/3triples/University.xml', 363,
             '<sentence ID="1"/>'): '<sentence ID="1"><striple>Romania | leaderTitle | Prime_Minister_of_Romania</striple><striple>Romania | leaderName | Klaus_Iohannis</striple></sentence><sentence ID="2"><striple>1_Decembrie_1918_University | country | Romania</striple></sentence>',
            ('dev/3triples/WrittenWork.xml', 1183, '<sentence ID="1"/>'): False,
            (
                'dev/3triples/WrittenWork.xml', 1184,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('dev/3triples/WrittenWork.xml', 1535, '<sentence ID="1"/>'): False,
            (
                'dev/3triples/WrittenWork.xml', 1536,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'dev/4triples/Airport.xml', 1295,
                '<sentence ID="1"/>'): '<sentence ID="1"><striple>Andrews_County_Airport | location | Texas</striple><striple>Texas | country | United_States</striple><striple>Texas | capital | Austin,_Texas</striple><striple>Texas | largestCity | Houston</striple></sentence>',
            ('dev/4triples/Building.xml', 2004, '<sentence ID="1"/>'): False, (
                'dev/4triples/Building.xml', 2005,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('dev/4triples/Monument.xml', 480, '<sentence ID="1"/>'): False, (
                'dev/4triples/Monument.xml', 481,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('dev/5triples/Airport.xml', 1188, '<sentence ID="1"/>'): False, (
                'dev/5triples/Airport.xml', 1189,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('dev/5triples/Monument.xml', 150, '<sentence ID="1"/>'): False, (
                'dev/5triples/Monument.xml', 151,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'dev/7triples/University.xml', 643,
                '<lex comment="good" lid="Id2">'): '<lex comment="bad" lid="Id2">',
            ('test/1triples/Building.xml', 467,
             '<reference entity="Julia_Morgan" number="1" tag="PATIENT-1" type="name">Julia Morgan</reference>'): '<reference entity="Julia_Morgan" number="1" tag="PATIENT-1" type="name">Julia Morgan</reference><reference entity="Asilomar_Conference_Grounds" number="2" tag="PATIENT-2" type="name">the grounds of Asilomar Conference</reference>\t\t\t',
            ('test/1triples/Building.xml', 470,
             '<template>PATIENT-1 was the architect of the grounds of Asilomar Conference .</template>'): '<template>PATIENT-1 was the architect of the grounds of Asilomar Conference .</template>',
            ('test/1triples/Building.xml', 799,
             '<reference entity="" number="2" tag="AGENT-1n" type="name">Ethiopian</reference>'): '<reference entity="Ethiopia" number="2" tag="AGENT-1" type="name">Ethiopian</reference>',
            ('test/1triples/Building.xml', 802,
             '<template>PATIENT-1 is an AGENT-1n leader .</template>'): '<template>PATIENT-1 is an AGENT-1 leader .</template>',
            ('test/1triples/Building.xml', 803,
             '<lexicalization>PATIENT-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=undefined] a AGENT-1n leader .</lexicalization>'): '<lexicalization>PATIENT-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=undefined] an AGENT-1 leader .</lexicalization>',
            (
                'test/1triples/Building.xml', 464,
                '<sentence ID="1"/>'): '<sentence ID="1"><striple>Asilomar_Conference_Grounds | architect | Julia_Morgan</striple></sentence>',
            ('test/1triples/Building.xml', 795,
             '<sentence ID="1"/>'): '<sentence ID="1"><striple>Ethiopia | leaderName | Mulatu_Teshome</striple></sentence>',
            ('test/1triples/ComicsCharacter.xml', 12,
             '<sentence ID="1"/>'): '<sentence ID="1"><striple>April_ONeil | creator | Kevin_Eastman</striple></sentence>',
            ('test/1triples/ComicsCharacter.xml', 12,
             '<sentence ID="1"/>'): '<sentence ID="1"><striple>April_ONeil | creator | Kevin_Eastman</striple></sentence>',
            ('test/1triples/Food.xml', 12,
             '<sentence ID="1"/>'): '<sentence ID="1"><striple>Amatriciana_sauce | ingredient | Pecorino_Romano</striple></sentence>',
            ('test/1triples/MeanOfTransportation.xml', 1015,
             '<otriple>Alhambra_(1855) | status | &quot;Wrecked&quot;</otriple>'): '<otriple>Alhambra_(1855) | status | Wrecked</otriple>',
            ('test/1triples/MeanOfTransportation.xml', 1018,
             '<mtriple>Alhambra | status | &quot;Wrecked&quot;</mtriple>'): '<mtriple>Alhambra | status | Wrecked</mtriple>',
            ('test/1triples/MeanOfTransportation.xml', 1022,
             '<sentence ID="1"/>'): '<sentence ID="1"><striple>Alhambra | status | Wrecked</striple></sentence>',
            ('test/1triples/Politician.xml', 1073,
             '<sentence ID="1"/>'): '<sentence ID="1"><striple>Albert_B._White | spouse | Agnes_Ward_White</striple></sentence>',
            ('test/1triples/Politician.xml', 1074, '<sentence ID="2"/>'): False,
            (
                'test/1triples/SportsTeam.xml', 962,
                '<sentence ID="1"/>'): '<sentence ID="1"><striple>Jorge_Humberto_Rodríguez | club | FC_Dallas</striple></sentence>',
            ('test/2triples/Airport.xml', 726,
             '<sentence ID="1"/>'): '<sentence ID="1"><striple>Amsterdam_Airport_Schiphol | elevationAboveTheSeaLevel_(in_metres) | -3.3528</striple><striple>Amsterdam_Airport_Schiphol | 1st_runway_SurfaceType | Asphalt</striple></sentence>',
            # ('test/2triples/Airport.xml', 727, '<sentence ID="2"/>'): False,
            (
                'test/2triples/Astronaut.xml', 14,
                '<sentence ID="1"/>'): '<sentence ID="1"><striple>Alan_Bean | birthDate | &quot;1932-03-15&quot;</striple><striple>Alan_Bean | status | &quot;Retired&quot;</striple></sentence>',
            ('test/3triples/Athlete.xml', 408,
             '<reference entity="United_Petrotrin_F.C." number="3" tag="BRIDGE-1" type="description">the United Petrotrin F.C . club .</reference>'): '<reference entity="United_Petrotrin_F.C." number="3" tag="BRIDGE-1" type="description">the United Petrotrin F.C. club</reference>',
            ('test/3triples/Athlete.xml', 411,
             '<text>Akeem Adams played for W Connection F.C. and is a member of the United Petrotrin F.C. club. which play in Palo Seco.</text>'): '<text>Akeem Adams played for W Connection F.C. and is a member of the United Petrotrin F.C. club, which play in Palo Seco.</text>',
            ('test/3triples/Athlete.xml', 2221,
             '<sentence ID="1"/>'): '<sentence ID="1"><striple>Abel_Hernández | club | Uruguay_Olympic_football_team</striple><striple>Abel_Hernández | club | Hull_City_A.F.C.</striple><striple>Hull_City_A.F.C. | manager | Steve_Bruce</striple></sentence>',
            ('test/3triples/CelestialBody.xml', 839,
             '<sentence ID="1"/>'): '<sentence ID="1"><striple>N._R._Pogson | nationality | England</striple><striple>N._R._Pogson | birthPlace | Nottingham</striple><striple>107_Camilla | discoverer | N._R._Pogson</striple></sentence>',
            ('test/3triples/SportsTeam.xml', 162,
             '<sentence ID="1"/>'): '<sentence ID="1"><striple>A.C._Lumezzane | fullname | &quot;Associazione Calcio Lumezzane SpA&quot;</striple><striple>A.C._Lumezzane | league | &quot;Lega Pro/A&quot;</striple><striple>A.C._Lumezzane | numberOfMembers | 4150</striple></sentence>',
            ('test/3triples/SportsTeam.xml', 163, '<sentence ID="2"/>'): False,
            ('test/3triples/SportsTeam.xml', 164, '<sentence ID="3"/>'): False,
            ('test/3triples/WrittenWork.xml', 793, '</sentence>'): False,
            ('test/3triples/WrittenWork.xml', 794, '<sentence ID="2">'): False,
            (
                'test/3triples/WrittenWork.xml', 805,
                '<text>Abh.Math.Semin.Univ.Hambg is the abbreviation for Abhandlungen aus dem Mathematischen Seminar der Universität Hamburg. which has the ISSN number 1865-8784 as well as the LCCN number 32024459.</text>'): '<text>Abh.Math.Semin.Univ.Hambg is the abbreviation for Abhandlungen aus dem Mathematischen Seminar der Universität Hamburg, which has the ISSN number 1865-8784 as well as the LCCN number 32024459.</text>',
            ('test/3triples/WrittenWork.xml', 806,
             '<template>PATIENT-3 is the abbreviation for AGENT-1 . which has the ISSN number PATIENT-1 as well as the LCCN number PATIENT-2 .</template>'): '<template>PATIENT-3 is the abbreviation for AGENT-1 , which has the ISSN number PATIENT-1 as well as the LCCN number PATIENT-2 .</template>',
            ('test/3triples/WrittenWork.xml', 807,
             '<lexicalization>PATIENT-3 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=defined] the abbreviation for AGENT-1 . which VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have DT[form=defined] the ISSN number PATIENT-1 as well as DT[form=defined] the LCCN number PATIENT-2 .</lexicalization>'): '<lexicalization>PATIENT-3 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=defined] the abbreviation for AGENT-1 , which VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have DT[form=defined] the ISSN number PATIENT-1 as well as DT[form=defined] the LCCN number PATIENT-2 .</lexicalization>',
            ('test/4triples/Athlete.xml', 844, '</sentence>'): False,
            ('test/4triples/Athlete.xml', 845, '<sentence ID="2">'): False, (
                'test/4triples/Athlete.xml', 857,
                "<text>Alaa Abdul Zahra, whose club is Al-Zawra'a SC, is also a member of the club, AL Kharaitiyat SC @ Amar Osim is the manager of Al Kharaitiyat SC. which is located in Al Khor.</text>"): "<text>Alaa Abdul Zahra, whose club is Al-Zawra'a SC, is also a member of the club, AL Kharaitiyat SC @ Amar Osim is the manager of Al Kharaitiyat SC, which is located in Al Khor.</text>",
            ('test/4triples/Athlete.xml', 858,
             '<template>AGENT-1 , whose club is PATIENT-2 , is also a member of the club , BRIDGE-1 @ PATIENT-3 is the manager of BRIDGE-1 . which is located in PATIENT-1 .</template>'): '<template>AGENT-1 , whose club is PATIENT-2 , is also a member of the club , BRIDGE-1 @ PATIENT-3 is the manager of BRIDGE-1 , which is located in PATIENT-1 .</template>',
            ('test/4triples/Athlete.xml', 859,
             '<lexicalization>AGENT-1 , whose club VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-2 , VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be also DT[form=undefined] a member of DT[form=defined] the club , BRIDGE-1 PATIENT-3 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=defined] the manager of BRIDGE-1 . which VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be located in PATIENT-1 .</lexicalization>'): '<lexicalization>AGENT-1 , whose club VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-2 , VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be also DT[form=undefined] a member of DT[form=defined] the club , BRIDGE-1 PATIENT-3 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=defined] the manager of BRIDGE-1 , which VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be located in PATIENT-1 .</lexicalization>',
            ('test/4triples/Athlete.xml', 1021, '</sentence>'): False,
            ('test/4triples/Athlete.xml', 1022, '<sentence ID="2">'): False, (
                'test/4triples/Athlete.xml', 1025,
                '<sentence ID="3">'): '<sentence ID="2">', (
                'test/4triples/Athlete.xml', 1037,
                '<text>Alaa Abdul-Zahra, whose club is Shabab Al-Ordon Club, also plays for Al Kharaitiyat SC. which is located in Al Khor. The manager of Al Kharaitiyat SC is Amar Osim.</text>'): '<text>Alaa Abdul-Zahra, whose club is Shabab Al-Ordon Club, also plays for Al Kharaitiyat SC, which is located in Al Khor. The manager of Al Kharaitiyat SC is Amar Osim.</text>',
            ('test/4triples/Athlete.xml', 1038,
             '<template>AGENT-1 , whose club is PATIENT-2 , also plays for BRIDGE-1 . which is located in PATIENT-1 . The manager of BRIDGE-1 is PATIENT-3 .</template>'): '<template>AGENT-1 , whose club is PATIENT-2 , also plays for BRIDGE-1 , which is located in PATIENT-1 . The manager of BRIDGE-1 is PATIENT-3 .</template>',
            ('test/4triples/Athlete.xml', 1039,
             '<lexicalization>AGENT-1 , whose club VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-2 , also VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] play for BRIDGE-1 . which VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be located in PATIENT-1 . DT[form=defined] the manager of BRIDGE-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-3 .</lexicalization>'): '<lexicalization>AGENT-1 , whose club VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-2 , also VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] play for BRIDGE-1 , which VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be located in PATIENT-1 . DT[form=defined] the manager of BRIDGE-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-3 .</lexicalization>',
            ('test/4triples/CelestialBody.xml', 34,
             '<text>The epoch of (19255) 1994 VK8 is on 31 December 2006. It has an orbital period of 8788850000.0, a periapsis of 6155910000000.0 and an apoapsis of 6603633000.0 km.</text>'): '<text>The epoch of (19255) 1994 VK8 is on 31 December 2006. It has an orbital period of 8788850000.0 and a periapsis of 6155910000000.0 .</text>',
            ('test/4triples/CelestialBody.xml', 35,
             '<template>The epoch of AGENT-1 is on PATIENT-1 . AGENT-1 has an orbital period of PATIENT-2 , a periapsis of PATIENT-3 and an apoapsis of PATIENT-5 .</template>'): '<template>The epoch of AGENT-1 is on PATIENT-1 . AGENT-1 has an orbital period of PATIENT-2 and a periapsis of PATIENT-3 .</template>',
            ('test/4triples/MeanOfTransportation.xml', 293,
             '<text>Costa Crociere is the owner of the AIDAstella which is 25326.0 millimetres long. It was built by Meyer Werft and operated by AIDA Cruise Line.</text>'): '<text>Costa Crociere is the owner of the AIDAstella which is 25326.0 millimetres long. It was built by Meyer Werft .</text>',
            ('test/4triples/MeanOfTransportation.xml', 294,
             '<template>PATIENT-4 is the owner of AGENT-1 which is PATIENT-2 long . AGENT-1 was built by PATIENT-3 and operated by BRIDGE-1 .</template>'): '<template>PATIENT-4 is the owner of AGENT-1 which is PATIENT-2 long . AGENT-1 was built by PATIENT-3 .</template>',
            ('test/4triples/MeanOfTransportation.xml', 3070,
             '<sentence ID="1"/>'): False, (
                'test/4triples/MeanOfTransportation.xml', 3071,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'test/4triples/Monument.xml', 381,
                '<text>Ahmet Davutoglu is the leader of Turkey where the capital is Ankara. The Ataturk monument (Izmir) which is made of bronze is located within the country.</text>'): '<text>Ahmet Davutoglu is the leader of Turkey where the capital is Ankara. The Ataturk monument (Izmir) is located within the country.</text>',
            ('test/4triples/Monument.xml', 382,
             '<template>PATIENT-1 is the leader of BRIDGE-1 where the capital is PATIENT-2 . AGENT-1 which is made of PATIENT-4 is located within BRIDGE-1 .</template>'): '<template>PATIENT-1 is the leader of BRIDGE-1 where the capital is PATIENT-2 . AGENT-1 is located within BRIDGE-1 .</template>',
            ('test/4triples/Politician.xml', 2169, '<sentence ID="1"/>'): False,
            (
                'test/4triples/Politician.xml', 2170,
                '<sentence ID="2">'): '<sentence ID="1">',
            (
                'test/4triples/WrittenWork.xml', 1315,
                '<sentence ID="1"/>'): False,
            (
                'test/4triples/WrittenWork.xml', 1316,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('test/5triples/Building.xml', 284, '<sentence ID="1"/>'): False, (
                'test/5triples/Building.xml', 285,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('test/5triples/Building.xml', 910, '</sentence>'): False,
            ('test/5triples/Building.xml', 911, '<sentence ID="2">'): False, (
                'test/5triples/Building.xml', 914,
                '<sentence ID="3">'): '<sentence ID="2">', (
                'test/5triples/CelestialBody.xml', 137,
                '<reference entity="101_Helena" number="6" tag="AGENT-1" type="pronoun">He</reference>'): False,
            ('test/5triples/CelestialBody.xml', 138,
             '<reference entity="Madison,_Wisconsin" number="7" tag="PATIENT-4" type="name">Madison , Wisconsin</reference>'): '<reference entity="Madison,_Wisconsin" number="6" tag="PATIENT-4" type="name">Madison , Wisconsin</reference>',
            ('test/5triples/CelestialBody.xml', 141,
             '<template>BRIDGE-1 , who discovered AGENT-1 on PATIENT-2 , is a PATIENT-3 national who attended PATIENT-1 . AGENT-1 died in PATIENT-4 .</template>'): '<template>BRIDGE-1 , who discovered AGENT-1 on PATIENT-2 , is a PATIENT-3 national who attended PATIENT-1 . BRIDGE-1 died in PATIENT-4 .</template>',
            ('test/5triples/CelestialBody.xml', 142,
             '<lexicalization>BRIDGE-1 , who VP[aspect=simple,tense=past,voice=active,person=null,number=null] discover AGENT-1 on PATIENT-2 , VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=undefined] a PATIENT-3 national who VP[aspect=simple,tense=past,voice=active,person=null,number=null] attend PATIENT-1 . AGENT-1 VP[aspect=simple,tense=past,voice=active,person=null,number=null] die in PATIENT-4 .</lexicalization>'):
                '<lexicalization>BRIDGE-1 , who VP[aspect=simple,tense=past,voice=active,person=null,number=null] discover AGENT-1 on PATIENT-2 , VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=undefined] a PATIENT-3 national who VP[aspect=simple,tense=past,voice=active,person=null,number=null] attend PATIENT-1 . BRIDGE-1 VP[aspect=simple,tense=past,voice=active,person=null,number=null] die in PATIENT-4 .</lexicalization>',
            ('test/5triples/CelestialBody.xml', 791,
             "<text>B. Zellner was the discoverer of 107 Camilla that has an orbital period of 2368.05 days. It's epoch is Dec. 31, 2006. The celestial body has a periapsis of 479343000.0 kilometres and an apoapsis of 560937000.0 km.</text>"): '<text>B. Zellner was the discoverer of 107 Camilla that has an orbital period of 2368.05 days. Its epoch is Dec. 31, 2006. Its celestial body has a periapsis of 479343000.0 kilometres and an apoapsis of 560937000.0 km.</text>',
            ('test/5triples/CelestialBody.xml', 792,
             '<template>PATIENT-1 was the discoverer of AGENT-1 that has an orbital period of PATIENT-2 . AGENT-1 epoch is PATIENT-4 . The celestial body has a periapsis of PATIENT-3 and an apoapsis of PATIENT-5 .</template>'): '<template>PATIENT-1 was the discoverer of AGENT-1 that has an orbital period of PATIENT-2 . PATIENT-1 epoch is PATIENT-4 . PATIENT-1 celestial body has a periapsis of PATIENT-3 and an apoapsis of PATIENT-5 .</template>',
            ('test/5triples/CelestialBody.xml', 793,
             '<lexicalization>PATIENT-1 VP[aspect=simple,tense=past,voice=active,person=null,number=singular] be DT[form=defined] the discoverer of AGENT-1 that VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have DT[form=undefined] a orbital period of PATIENT-2 . AGENT-1 epoch VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-4 . DT[form=defined] the celestial body VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have DT[form=undefined] a periapsis of PATIENT-3 and DT[form=undefined] a apoapsis of PATIENT-5 .</lexicalization>'): '<lexicalization>PATIENT-1 VP[aspect=simple,tense=past,voice=active,person=null,number=singular] be DT[form=defined] the discoverer of AGENT-1 that VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have DT[form=undefined] a orbital period of PATIENT-2 . PATIENT-1 epoch VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-4 . PATIENT-1 celestial body VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have DT[form=undefined] a periapsis of PATIENT-3 and DT[form=undefined] a apoapsis of PATIENT-5 .</lexicalization>',
            ('test/5triples/CelestialBody.xml', 888,
             '<text>107 Camilla, epoch date 31 December 2006, was discovered by C Woods and has an orbital period of 2368.05 days. The apoapsis and periapsis measurements are 560937000.0 km and 479343000.0 km respectively.</text>'): "<text>107 Camilla, epoch date 31 December 2006, was discovered by C Woods and has an orbital period of 2368.05 days. 107 Camilla's apoapsis and periapsis measurements are 560937000.0 km and 479343000.0 km respectively.</text>",
            ('test/5triples/CelestialBody.xml', 889,
             '<template>AGENT-1 , epoch date PATIENT-4 , was discovered by PATIENT-1 and has an orbital period of PATIENT-2 . The apoapsis and periapsis measurements are PATIENT-5 and PATIENT-3 respectively .</template>'): '<template>AGENT-1 , epoch date PATIENT-4 , was discovered by PATIENT-1 and has an orbital period of PATIENT-2 . AGENT-1 apoapsis and periapsis measurements are PATIENT-5 and PATIENT-3 respectively .</template>',
            ('test/5triples/CelestialBody.xml', 890,
             '<lexicalization>AGENT-1 , epoch date PATIENT-4 , VP[aspect=simple,tense=past,voice=passive,person=null,number=singular] discover by PATIENT-1 and VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have DT[form=undefined] a orbital period of PATIENT-2 . DT[form=defined] the apoapsis and periapsis measurements VP[aspect=simple,tense=present,voice=active,person=non-3rd,number=plural] be PATIENT-5 and PATIENT-3 respectively .</lexicalization>'): '<lexicalization>AGENT-1 , epoch date PATIENT-4 , VP[aspect=simple,tense=past,voice=passive,person=null,number=singular] discover by PATIENT-1 and VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have DT[form=undefined] a orbital period of PATIENT-2 . AGENT-1 apoapsis and periapsis measurements VP[aspect=simple,tense=present,voice=active,person=non-3rd,number=plural] be PATIENT-5 and PATIENT-3 respectively .</lexicalization>',
            ('test/5triples/CelestialBody.xml', 961,
             '<text>107 Camilla, which has the epoch date 31 December 2006, was discovered by F Vilas and has an orbital period of 2368.05 days. The apoapsis and periapsis measurements are 560937000.0 kilometres and 479343000.0 kilometres respectively.</text>'): "<text>107 Camilla, which has the epoch date 31 December 2006, was discovered by F Vilas and has an orbital period of 2368.05 days. 107 Camilla's apoapsis and periapsis measurements are 560937000.0 kilometres and 479343000.0 kilometres respectively.</text>",
            ('test/5triples/CelestialBody.xml', 962,
             '<template>AGENT-1 , which has the epoch date PATIENT-2 , was discovered by PATIENT-1 and has an orbital period of PATIENT-3 . The apoapsis and periapsis measurements are PATIENT-5 and PATIENT-4 respectively .</template>'): '<template>AGENT-1 , which has the epoch date PATIENT-2 , was discovered by PATIENT-1 and has an orbital period of PATIENT-3 . AGENT-1 apoapsis and periapsis measurements are PATIENT-5 and PATIENT-4 respectively .</template>',
            ('test/5triples/CelestialBody.xml', 963,
             '<lexicalization>AGENT-1 , which VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have DT[form=defined] the epoch date PATIENT-2 , VP[aspect=simple,tense=past,voice=passive,person=null,number=singular] discover by PATIENT-1 and VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have DT[form=undefined] a orbital period of PATIENT-3 . DT[form=defined] the apoapsis and periapsis measurements VP[aspect=simple,tense=present,voice=active,person=non-3rd,number=plural] be PATIENT-5 and PATIENT-4 respectively .</lexicalization>'): '<lexicalization>AGENT-1 , which VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have DT[form=defined] the epoch date PATIENT-2 , VP[aspect=simple,tense=past,voice=passive,person=null,number=singular] discover by PATIENT-1 and VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have DT[form=undefined] a orbital period of PATIENT-3 . AGENT-1 apoapsis and periapsis measurements VP[aspect=simple,tense=present,voice=active,person=non-3rd,number=plural] be PATIENT-5 and PATIENT-4 respectively .</lexicalization>',
            ('test/5triples/CelestialBody.xml', 1357,
             '<text>11264 Claudiomaccone has an epoch date of November 26th 2005, an orbital period of 1513.722 days. a periapsis of 296521000.0 km, an apoapsis of 475426000.0 km, and a temperature of 173.0 kelvins.</text>'): '<text>11264 Claudiomaccone has an epoch date of November 26th 2005, an orbital period of 1513.722 days. It has a periapsis of 296521000.0 km, an apoapsis of 475426000.0 km, and a temperature of 173.0 kelvins.</text>',
            ('test/5triples/CelestialBody.xml', 1358,
             '<template>AGENT-1 has an epoch date of PATIENT-1 , an orbital period of PATIENT-2 . a periapsis of PATIENT-3 , an apoapsis of PATIENT-4 , and a temperature of PATIENT-5 .</template>'): '<template>AGENT-1 has an epoch date of PATIENT-1 , an orbital period of PATIENT-2 . AGENT-1 has a periapsis of PATIENT-3 , an apoapsis of PATIENT-4 , and a temperature of PATIENT-5 .</template>',
            ('test/5triples/CelestialBody.xml', 1359,
             '<lexicalization>AGENT-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have DT[form=undefined] a epoch date of PATIENT-1 , DT[form=undefined] a orbital period of PATIENT-2 . DT[form=undefined] a periapsis of PATIENT-3 , DT[form=undefined] a apoapsis of PATIENT-4 , and DT[form=undefined] a temperature of PATIENT-5 .</lexicalization>'): '<lexicalization>AGENT-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have DT[form=undefined] a epoch date of PATIENT-1 , DT[form=undefined] a orbital period of PATIENT-2 . AGENT-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have DT[form=undefined] a periapsis of PATIENT-3 , DT[form=undefined] a apoapsis of PATIENT-4 , and DT[form=undefined] a temperature of PATIENT-5 .</lexicalization>',
            ('test/5triples/Food.xml', 20, '<sentence ID="1"/>'): False,
            ('test/5triples/Food.xml', 21,
             '<sentence ID="2">'): '<sentence ID="1">',
            ('test/5triples/MeanOfTransportation.xml', 131,
             '<reference entity="" number="3" tag="BRIDGE-1" type="name">AIDA Cruise Line</reference>'): '<reference entity="AIDA_Cruises" number="3" tag="BRIDGE-1" type="name">AIDA Cruise Line</reference>',
            ('test/5triples/MeanOfTransportation.xml', 137,
             '<text>AIDAstella was built by Meyer Werft and is operated by AIDA Cruise Line. The AIDAstella has a beam of 32.2 m, is 253260.0 millimetres in length and has a beam of 32.2 m.</text>'): '<text>AIDAstella was built by Meyer Werft. The AIDAstella has a beam of 32.2 m, is 253260.0 millimetres in length and has a beam of 32.2 m.</text>',
            ('test/5triples/MeanOfTransportation.xml', 121,
             '<striple>AIDAstella | builder | Meyer_Werft</striple>'): '<striple>AIDAstella | builder | Meyer_Werft</striple><striple>AIDAstella | operator | AIDA_Cruises</striple>',
            ('test/5triples/MeanOfTransportation.xml', 138,
             '<template>AGENT-1 was built by PATIENT-4 and is operated by BRIDGE-1 . AGENT-1 has a beam of PATIENT-2 , is PATIENT-5 in length and has a beam of PATIENT-2 .</template>'): '<template>AGENT-1 was built by PATIENT-4 . AGENT-1 has a beam of PATIENT-2 , is PATIENT-5 in length and has a beam of PATIENT-2 .</template>',
            ('test/5triples/MeanOfTransportation.xml', 161,
             '<text>The AIDAstella was built by Meyer Werft and operated by the AIDA Cruise Line. It is 253260.0 millimetres long with a beam of 32.2 metres and a top speed of 38.892 km/h.</text>'): '<text>The AIDAstella was built by Meyer Werft. It is 253260.0 millimetres long with a beam of 32.2 metres and a top speed of 38.892 km/h.</text>',
            ('test/5triples/MeanOfTransportation.xml', 162,
             '<template>AGENT-1 was built by PATIENT-4 and operated by BRIDGE-1 . AGENT-1 is PATIENT-5 long with a beam of PATIENT-2 and a top speed of PATIENT-3 .</template>'): '<template>AGENT-1 was built by PATIENT-4 . AGENT-1 is PATIENT-5 long with a beam of PATIENT-2 and a top speed of PATIENT-3 .</template>',
            ('test/5triples/Monument.xml', 387,
             '<sentence ID="1"/>'): '<sentence ID="1">',
            ('test/5triples/Monument.xml', 388, '<sentence ID="2">'): False, (
                'test/5triples/Monument.xml', 394,
                '<sentence ID="3">'): '<sentence ID="2">',
            ('test/5triples/WrittenWork.xml', 121, '<sentence ID="1"/>'): False,
            (
                'test/5triples/WrittenWork.xml', 122,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'test/5triples/WrittenWork.xml', 127,
                '<sentence ID="3">'): '<sentence ID="2">', (
                'test/5triples/WrittenWork.xml', 140,
                '<text>Affiliated with the Association of Public and Land grant Universities and the Association of American Universities., Cornell University is the publisher of the Administrative Science Quarterly. The university is located in Ithaca New York and the president is Elizabeth Garrett.</text>'): '<text>Affiliated with the Association of Public and Land grant Universities and the Association of American Universities, Cornell University is the publisher of the Administrative Science Quarterly. The university is located in Ithaca New York and the president is Elizabeth Garrett.</text>',
            ('test/5triples/WrittenWork.xml', 141,
             '<template>Affiliated with PATIENT-1 and PATIENT-2 . , BRIDGE-1 is the publisher of AGENT-1 . BRIDGE-1 is located in PATIENT-4 and the president is PATIENT-3 .</template>'): '<template>Affiliated with PATIENT-1 and PATIENT-2 , BRIDGE-1 is the publisher of AGENT-1 . BRIDGE-1 is located in PATIENT-4 and the president is PATIENT-3 .</template>',
            (
                'test/5triples/WrittenWork.xml', 142,
                '<lexicalization>VP[aspect=simple,tense=past,voice=active,person=null,number=null] affiliate with PATIENT-1 and PATIENT-2 . , BRIDGE-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=defined] the publisher of AGENT-1 . BRIDGE-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be located in PATIENT-4 and DT[form=defined] the president VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-3 .</lexicalization>'):
                '<lexicalization>VP[aspect=simple,tense=past,voice=active,person=null,number=null] affiliate with PATIENT-1 and PATIENT-2 , BRIDGE-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=defined] the publisher of AGENT-1 . BRIDGE-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be located in PATIENT-4 and DT[form=defined] the president VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-3 .</lexicalization>',
            ('test/6triples/Astronaut.xml', 461,
             '<text>Buzz Aldrin was born on 20th January 1930 in Glen Ridge New Jersey. He graduated from MIT in 1963 and was a member of the Apollo 11 crew, operated by NASA. The back up pilot was William Anders.</text>'): '<text>Buzz Aldrin was born on 20th January 1930 in Glen Ridge New Jersey. He graduated from MIT in 1963 and was a member of the Apollo 11 crew, operated by NASA. The back up pilot of Apollo 11 was William Anders.</text>',
            ('test/6triples/Astronaut.xml', 462,
             '<template>AGENT-1 was born on PATIENT-2 in PATIENT-1 . AGENT-1 graduated from PATIENT-3 and was a member of the BRIDGE-1 crew , operated by PATIENT-5 . The back up pilot was PATIENT-4 .</template>'): '<template>AGENT-1 was born on PATIENT-2 in PATIENT-1 . AGENT-1 graduated from PATIENT-3 and was a member of the BRIDGE-1 crew , operated by PATIENT-5 . The back up pilot of BRIDGE-1 was PATIENT-4 .</template>',
            ('test/6triples/Astronaut.xml', 463,
             '<lexicalization>AGENT-1 VP[aspect=simple,tense=past,voice=passive,person=null,number=singular] bear on PATIENT-2 in PATIENT-1 . AGENT-1 VP[aspect=simple,tense=past,voice=active,person=null,number=null] graduate from PATIENT-3 and VP[aspect=simple,tense=past,voice=active,person=null,number=singular] be DT[form=undefined] a member of DT[form=defined] the BRIDGE-1 crew , VP[aspect=simple,tense=past,voice=active,person=null,number=null] operate by PATIENT-5 . DT[form=defined] the back up pilot VP[aspect=simple,tense=past,voice=active,person=null,number=singular] be PATIENT-4 .</lexicalization>'): '<lexicalization>AGENT-1 VP[aspect=simple,tense=past,voice=passive,person=null,number=singular] bear on PATIENT-2 in PATIENT-1 . AGENT-1 VP[aspect=simple,tense=past,voice=active,person=null,number=null] graduate from PATIENT-3 and VP[aspect=simple,tense=past,voice=active,person=null,number=singular] be DT[form=undefined] a member of DT[form=defined] the BRIDGE-1 crew , VP[aspect=simple,tense=past,voice=active,person=null,number=null] operate by PATIENT-5 . DT[form=defined] the back up pilot of BRIDGE-1 VP[aspect=simple,tense=past,voice=active,person=null,number=singular] be PATIENT-4 .</lexicalization>',
            ('test/6triples/Astronaut.xml', 815,
             '<reference entity="William_Anders" number="1" tag="AGENT-1" type="name">William Anders</reference>'): '<reference entity="William_Anders" number="1" tag="AGENT-1" type="name">William Anders</reference><reference entity="United_States" number="2" tag="PATIENT-2" type="name">American</reference><reference entity="Fighter_pilot" number="3" tag="PATIENT-3" type="description">fighter pilot</reference>',
            ('test/6triples/Astronaut.xml', 958,
             '<reference entity="Frank_Borman" number="7" tag="PATIENT-3" type="name">Frank Borman</reference>'): '<reference entity="Buzz_Aldrin" number="6" tag="PATIENT-2" type="name">Buzz Aldrin</reference><reference entity="Frank_Borman" number="7" tag="PATIENT-3" type="name">Frank Borman</reference>',

            ('test/6triples/Astronaut.xml', 960,
             "<text>William Anders, retired, was a member of NASA's @ Apollo 8 after graduating from AFIT in 1962 with an MS. Buzz Aldrin was a back up pilot and Frank Borman a crew member.</text>"): "<text>William Anders, retired, was a member of NASA's @ Apollo 8 after graduating from AFIT in 1962 with an MS. Buzz Aldrin was a back up pilot of Apollo 8 and Frank Borman a crew member.</text>",
            ('test/6triples/Astronaut.xml', 961,
             '<template>AGENT-1 , PATIENT-5 , was a member of PATIENT-4 @ BRIDGE-1 after graduating from PATIENT-1 . PATIENT-2 was a back up pilot and PATIENT-3 a crew member .</template>'): '<template>AGENT-1 , PATIENT-5 , was a member of PATIENT-4 @ BRIDGE-1 after graduating from PATIENT-1 . PATIENT-2 was a back up pilot of BRIDGE-1 and PATIENT-3 a crew member .</template>',
            ('test/6triples/Astronaut.xml', 962,
             '<lexicalization>AGENT-1 , PATIENT-5 , VP[aspect=simple,tense=past,voice=active,person=null,number=singular] be DT[form=undefined] a member of PATIENT-4 BRIDGE-1 after VP[aspect=progressive,tense=present,voice=active,person=null,number=null] graduate from PATIENT-1 . PATIENT-2 VP[aspect=simple,tense=past,voice=active,person=null,number=singular] be DT[form=undefined] a back up pilot and PATIENT-3 DT[form=undefined] a crew member .</lexicalization>'): '<lexicalization>AGENT-1 , PATIENT-5 , VP[aspect=simple,tense=past,voice=active,person=null,number=singular] be DT[form=undefined] a member of PATIENT-4 BRIDGE-1 after VP[aspect=progressive,tense=present,voice=active,person=null,number=null] graduate from PATIENT-1 . PATIENT-2 VP[aspect=simple,tense=past,voice=active,person=null,number=singular] be DT[form=undefined] a back up pilot of BRDIGE-1 and PATIENT-3 DT[form=undefined] a crew member .</lexicalization>',
            ('test/6triples/Monument.xml', 135, '</sentence>'): False,
            ('test/6triples/Monument.xml', 136, '<sentence ID="2">'): False, (
                'test/6triples/Monument.xml', 140,
                '<sentence ID="3">'): '<sentence ID="2">', (
                'test/6triples/Monument.xml', 144,
                '<sentence ID="4">'): '<sentence ID="3">', (
                'test/6triples/Monument.xml', 158,
                '<text>The location of the 11th Mississippi Infantry Monument is in Adams County, Pennsylvania. which has Franklin County to the west and Carroll County Maryland to the southeast. Cumberland County lies to the north with Frederick County, Maryland to the southwest. The 11th Mississippi Infantry Monument is a contributing property.</text>'): '<text>The location of the 11th Mississippi Infantry Monument is in Adams County, Pennsylvania, which has Franklin County to the west and Carroll County Maryland to the southeast. Cumberland County lies to the north with Frederick County, Maryland to the southwest. The 11th Mississippi Infantry Monument is a contributing property.</text>',
            ('test/6triples/Monument.xml', 159,
             '<template>The location of AGENT-1 is in BRIDGE-1 . which has PATIENT-1 to the west and PATIENT-2 to the southeast . PATIENT-3 lies to the north with PATIENT-4 to the southwest . AGENT-1 is PATIENT-5 .</template>'): '<template>The location of AGENT-1 is in BRIDGE-1 , which has PATIENT-1 to the west and PATIENT-2 to the southeast . PATIENT-3 lies to the north with PATIENT-4 to the southwest . AGENT-1 is PATIENT-5 .</template>',
            ('test/6triples/Monument.xml', 160,
             '<lexicalization>DT[form=defined] the location of AGENT-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be in BRIDGE-1 . which VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have PATIENT-1 to DT[form=defined] the west and PATIENT-2 to DT[form=defined] the southeast . PATIENT-3 VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] lie to DT[form=defined] the north with PATIENT-4 to DT[form=defined] the southwest . AGENT-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-5 .</lexicalization>'): '<lexicalization>DT[form=defined] the location of AGENT-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be in BRIDGE-1 , which VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have PATIENT-1 to DT[form=defined] the west and PATIENT-2 to DT[form=defined] the southeast . PATIENT-3 VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] lie to DT[form=defined] the north with PATIENT-4 to DT[form=defined] the southwest . AGENT-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-5 .</lexicalization>',
            ('test/6triples/University.xml', 99,
             '<text>The Accademia Di Architettura di Mendrisio is located in the city of Mendrisio, region Ticino in Switzerland. It was founded in 1996 and the dean is Mario Botta. There is currently 100 members of staff.</text>'): '<text>The Accademia Di Architettura di Mendrisio is located in the city of Mendrisio, region Ticino in Switzerland. It was founded in 1996 and the dean is Mario Botta. There is currently 100 members of staff in the Accademia Di Architettura di Mendrisio.</text>',
            ('test/6triples/University.xml', 100,
             '<template>AGENT-1 is located in the city of PATIENT-3 , region PATIENT-6 in PATIENT-1 . AGENT-1 was founded in PATIENT-4 and the dean is PATIENT-2 . There is currently PATIENT-5 members of staff .</template>'): '<template>AGENT-1 is located in the city of PATIENT-3 , region PATIENT-6 in PATIENT-1 . AGENT-1 was founded in PATIENT-4 and the dean is PATIENT-2 . There is currently PATIENT-5 members of staff in AGENT-1 .</template>',
            ('test/6triples/University.xml', 101,
             '<lexicalization>AGENT-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be located in DT[form=defined] the city of PATIENT-3 , region PATIENT-6 in PATIENT-1 . AGENT-1 VP[aspect=simple,tense=past,voice=passive,person=null,number=singular] found in PATIENT-4 and DT[form=defined] the dean VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-2 . There VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be currently PATIENT-5 members of staff .</lexicalization>'):
                '<lexicalization>AGENT-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be located in DT[form=defined] the city of PATIENT-3 , region PATIENT-6 in PATIENT-1 . AGENT-1 VP[aspect=simple,tense=past,voice=passive,person=null,number=singular] found in PATIENT-4 and DT[form=defined] the dean VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-2 . There VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be currently PATIENT-5 members of staff in AGENT-1.</lexicalization>',
            ('test/6triples/University.xml', 601,
             "<text>The 1 Decembrie 1918 University is located in Romania. Romania's capital is Bucharest; its leader is Klaus Iohannis and its patron saint is Andrew the Apostle. The ethnic group is the Germans of Romania and the anthem is Desteapta-te, romane!</text>"): "<text>The 1 Decembrie 1918 University is located in Romania. Romania's capital is Bucharest; its leader is Klaus Iohannis and its patron saint is Andrew the Apostle. Romania's ethnic group is the Germans of Romania and the anthem is Desteapta-te, romane!</text>",
            ('test/6triples/University.xml', 602,
             '<template>AGENT-1 is located in BRIDGE-1 . BRIDGE-1 capital is PATIENT-4 ; BRIDGE-1 leader is PATIENT-2 and BRIDGE-1 patron saint is PATIENT-3 . The ethnic group is PATIENT-1 and the anthem is PATIENT-5</template>'): '<template>AGENT-1 is located in BRIDGE-1 . BRIDGE-1 capital is PATIENT-4 ; BRIDGE-1 leader is PATIENT-2 and BRIDGE-1 patron saint is PATIENT-3 . BRIDGE-1 ethnic group is PATIENT-1 and the anthem is PATIENT-5</template>',
            ('test/6triples/University.xml', 603,
             '<lexicalization>AGENT-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be located in BRIDGE-1 . BRIDGE-1 capital VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-4 ; BRIDGE-1 leader VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-2 and BRIDGE-1 patron saint VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-3 . DT[form=defined] the ethnic group VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-1 and DT[form=defined] the anthem VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-5</lexicalization>'): '<lexicalization>AGENT-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be located in BRIDGE-1 . BRIDGE-1 capital VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-4 ; BRIDGE-1 leader VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-2 and BRIDGE-1 patron saint VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-3 . AGENT-1 ethnic group VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-1 and DT[form=defined] the anthem VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be PATIENT-5</lexicalization>',
            ('test/7triples/Astronaut.xml', 437, '<sentence ID="1"/>'): False, (
                'test/7triples/Astronaut.xml', 438,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('train/2triples/Airport.xml', 1295, '<sentence ID="1"/>'): False, (
                'train/2triples/Airport.xml', 1296,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('train/2triples/Airport.xml', 2417, '<sentence ID="1"/>'): False, (
                'train/2triples/Airport.xml', 2418,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/2triples/Food.xml', 12096,
                '<sentence ID="1"/>'): '<sentence ID="1"><striple>Bhajji | alternativeName | &quot;Bhaji, bajji&quot;</striple><striple>Bhajji | ingredient | Gram_flour</striple></sentence>',
            ('train/2triples/Food.xml', 12171,
             '<sentence ID="1"/>'): '<sentence ID="1"><striple>Bhajji | alternativeName | &quot;Bhaji, bajji&quot;</striple><striple>Bhajji | ingredient | Vegetable</striple></sentence>',
            ('train/3triples/Airport.xml', 7235,
             '<sentence ID="1"/>'): '<sentence ID="1"><striple>Andrews_County_Airport | location | Texas</striple><striple>Texas | capital | Austin,_Texas</striple><striple>Texas | language | Spanish_language</striple></sentence>',
            ('train/3triples/Airport.xml', 10400, '<sentence ID="1"/>'): False,
            (
                'train/3triples/Airport.xml', 10401,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/3triples/Airport.xml', 11543,
                '<sentence ID="1"/>'): '<sentence ID="1"><striple>Adolfo_Suárez_Madrid–Barajas_Airport | location | Madrid</striple><striple>Madrid | isPartOf | Community_of_Madrid</striple><striple>Madrid | country | Spain</striple></sentence>',
            ('train/3triples/Building.xml', 6267,
             '<sentence ID="1"/>'): '<sentence ID="1"><striple>Addis_Ababa | isPartOf | Addis_Ababa_Stadium</striple><striple>Addis_Ababa_City_Hall | location | Addis_Ababa</striple><striple>Addis_Ababa | country | Ethiopia</striple></sentence>',
            ('train/3triples/Building.xml', 13073, '<sentence ID="1"/>'): False,
            (
                'train/3triples/Building.xml', 13074,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/3triples/ComicsCharacter.xml', 1213,
                '<sentence ID="1"/>'): False,
            ('train/3triples/ComicsCharacter.xml', 1214,
             '<sentence ID="2">'): '<sentence ID="1">', (
                'train/3triples/ComicsCharacter.xml', 3027,
                '<sentence ID="1"/>'): False,
            ('train/3triples/ComicsCharacter.xml', 3028,
             '<sentence ID="2">'): '<sentence ID="1">', (
                'train/3triples/ComicsCharacter.xml', 3100,
                '<sentence ID="1"/>'): False,
            ('train/3triples/ComicsCharacter.xml', 3101,
             '<sentence ID="2">'): '<sentence ID="1">', (
                'train/3triples/Food.xml', 2730,
                '<sentence ID="1"/>'): '<sentence ID="1"><striple>Tomato | family | Solanaceae</striple><striple>Arrabbiata_sauce | ingredient | Tomato</striple></sentence>',
            ('train/3triples/Food.xml', 10588, '<sentence ID="1"/>'): False, (
                'train/3triples/Food.xml', 10589,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/3triples/Food.xml', 14354,
                '<sentence ID="1"/>'): '<sentence ID="1"><striple>Bhajji | country | India</striple><striple>Bhajji | mainIngredients | &quot;Gram flour, vegetables&quot;</striple></sentence>',
            ('train/3triples/Monument.xml', 626, '<sentence ID="1"/>'): False,
            ('train/3triples/Monument.xml', 627, '<sentence ID="2"/>'): False, (
                'train/3triples/Monument.xml', 628,
                '<sentence ID="3">'): '<sentence ID="1">',
            ('train/3triples/Monument.xml', 2649, '<sentence ID="1"/>'): False,
            ('train/3triples/Monument.xml', 2650,
             '<sentence ID="2">'): '<sentence ID="1">',
            (
                'train/3triples/SportsTeam.xml', 1784,
                '<sentence ID="1"/>'): '<sentence ID="1"><striple>A.C._Chievo_Verona | fullname | &quot;Associazione Calcio ChievoVerona S.r.l. &quot;</striple><striple>A.C._Chievo_Verona | ground | Verona</striple><striple>A.C._Chievo_Verona | numberOfMembers | 39371</striple></sentence>',
            ('train/3triples/SportsTeam.xml', 1816,
             '<sentence ID="1"/>'): '<sentence ID="1"><striple>A.C._Chievo_Verona | fullname | &quot;Associazione Calcio ChievoVerona S.r.l. &quot;</striple><striple>A.C._Chievo_Verona | ground | Verona</striple><striple>A.C._Chievo_Verona | numberOfMembers | 39371</striple></sentence>',
            (
                'train/3triples/SportsTeam.xml', 8837,
                '<sentence ID="1"/>'): False,
            (
                'train/3triples/SportsTeam.xml', 8838,
                '<sentence ID="2">'): '<sentence ID="1">',
            (
                'train/3triples/SportsTeam.xml', 9413,
                '<sentence ID="1"/>'): False,
            (
                'train/3triples/SportsTeam.xml', 9414,
                '<sentence ID="2">'): '<sentence ID="1">',
            (
                'train/3triples/University.xml', 1075,
                '<sentence ID="1"/>'): False,
            (
                'train/3triples/University.xml', 1076,
                '<sentence ID="2">'): '<sentence ID="1">',
            (
                'train/3triples/University.xml', 1195,
                '<sentence ID="1"/>'): False,
            (
                'train/3triples/University.xml', 1196,
                '<sentence ID="2">'): '<sentence ID="1">',
            (
                'train/3triples/University.xml', 2961,
                '<sentence ID="1"/>'): False,
            (
                'train/3triples/University.xml', 2962,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('train/3triples/WrittenWork.xml', 1874,
             '<sentence ID="1"/>'): False, (
                'train/3triples/WrittenWork.xml', 1875,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/4triples/Airport.xml', 15273,
                '<sentence ID="1"/>'): '<sentence ID="1"><striple>Infraero | location | Brasília</striple></sentence>',
            ('train/4triples/Airport.xml', 15280,
             '<striple>Infraero | location | Brasília</striple>'): False,
            ('train/4triples/Building.xml', 13568,
             '<reference entity="" number="2" tag="PATIENT-q" type="name">area B</reference>'): '<reference entity="B_postcode_area" number="2" tag="PATIENT-1" type="name">area B</reference>',
            ('train/4triples/Building.xml', 13573,
             '<template>BRIDGE-2 ( postcode PATIENT-q) , is home to BRIDGE-1 , the architect who designed AGENT-1 .</template>'): '<template>BRIDGE-2 ( postcode PATIENT-1) , is home to BRIDGE-1 , the architect who designed AGENT-1 .</template>',
            ('train/4triples/Building.xml', 14858, '<sentence ID="1"/>'): False,
            (
                'train/4triples/Building.xml', 14859,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('train/4triples/Building.xml', 15624, '<sentence ID="1"/>'): False,
            (
                'train/4triples/Building.xml', 15625,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('train/4triples/Food.xml', 3094, '<sentence ID="1"/>'): False, (
                'train/4triples/Food.xml', 3095,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('train/4triples/Food.xml', 7768, '<sentence ID="1"/>'): False, (
                'train/4triples/Food.xml', 7769,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/4triples/Food.xml', 7773,
                '<sentence ID="3">'): '<sentence ID="2">', (
                'train/4triples/Food.xml', 7786,
                "<text>The bacon sandwich. which is found in the UK, has different names including: Bacon butty, bacon sarnie, rasher sandwich, bacon sanger, piece 'n bacon, bacon cob, bacon barm and bacon muffin. Bread is an ingredient of this sandwich, which is a variation on a BLT.</text>"): "<text>The bacon sandwich, which is found in the UK, has different names including: Bacon butty, bacon sarnie, rasher sandwich, bacon sanger, piece 'n bacon, bacon cob, bacon barm and bacon muffin. Bread is an ingredient of this sandwich, which is a variation on a BLT.</text>",
            ('train/4triples/Food.xml', 7787,
             '<template>AGENT-1 . which is found in PATIENT-2 , has different names including : PATIENT-3 . PATIENT-4 is an ingredient of AGENT-1 , which is a variation on PATIENT-1 .</template>'): '<template>AGENT-1 , which is found in PATIENT-2 , has different names including : PATIENT-3 . PATIENT-4 is an ingredient of AGENT-1 , which is a variation on PATIENT-1 .</template>',
            ('train/4triples/Food.xml', 7788,
             '<lexicalization>AGENT-1 . which VP[aspect=simple,tense=present,voice=passive,person=3rd,number=singular] find in PATIENT-2 , VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have different names VP[aspect=progressive,tense=present,voice=active,person=null,number=null] include : PATIENT-3 . PATIENT-4 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=undefined] a ingredient of AGENT-1 , which VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=undefined] a variation on PATIENT-1 .</lexicalization>'): '<lexicalization>AGENT-1 , which VP[aspect=simple,tense=present,voice=passive,person=3rd,number=singular] find in PATIENT-2 , VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have different names VP[aspect=progressive,tense=present,voice=active,person=null,number=null] include : PATIENT-3 . PATIENT-4 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=undefined] a ingredient of AGENT-1 , which VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=undefined] a variation on PATIENT-1 .</lexicalization>',
            ('train/4triples/Food.xml', 8715, '<sentence ID="1"/>'): False, (
                'train/4triples/Food.xml', 8716,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('train/4triples/Monument.xml', 2551, '<sentence ID="1"/>'): False,
            (
                'train/4triples/Monument.xml', 2552,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('train/4triples/Monument.xml', 3877, '<sentence ID="1"/>'): False,
            (
                'train/4triples/Monument.xml', 3878,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('train/4triples/Monument.xml', 3919, '<sentence ID="1"/>'): False,
            (
                'train/4triples/Monument.xml', 3920,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/4triples/SportsTeam.xml', 9061,
                '<sentence ID="2">'): '<sentence ID="1">',
            (
                'train/4triples/SportsTeam.xml', 9060,
                '<sentence ID="1"/>'): False,
            ('train/4triples/WrittenWork.xml', 2012,
             '<sentence ID="1"/>'): False, (
                'train/4triples/WrittenWork.xml', 2013,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/4triples/WrittenWork.xml', 2017,
                '<sentence ID="3">'): '<sentence ID="2">',
            ('train/4triples/WrittenWork.xml', 2184,
             '<sentence ID="1"/>'): False, (
                'train/4triples/WrittenWork.xml', 2185,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/4triples/WrittenWork.xml', 2189,
                '<sentence ID="3">'): '<sentence ID="2">',
            ('train/4triples/WrittenWork.xml', 6624,
             '<sentence ID="1"/>'): False, (
                'train/4triples/WrittenWork.xml', 6625,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/4triples/WrittenWork.xml', 6629,
                '<sentence ID="3">'): '<sentence ID="2">',
            ('train/4triples/WrittenWork.xml', 6901,
             '<sentence ID="1"/>'): False, (
                'train/4triples/WrittenWork.xml', 6902,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/4triples/WrittenWork.xml', 6906,
                '<sentence ID="3">'): '<sentence ID="2">',
            ('train/4triples/WrittenWork.xml', 6971,
             '<sentence ID="1"/>'): False, (
                'train/4triples/WrittenWork.xml', 6972,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/4triples/WrittenWork.xml', 6976,
                '<sentence ID="3">'): '<sentence ID="2">',
            ('train/4triples/WrittenWork.xml', 7015,
             '<sentence ID="1"/>'): False, (
                'train/4triples/WrittenWork.xml', 7016,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('train/4triples/WrittenWork.xml', 7147,
             '<sentence ID="1"/>'): False, (
                'train/4triples/WrittenWork.xml', 7148,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/4triples/WrittenWork.xml', 7152,
                '<sentence ID="3">'): '<sentence ID="2">',
            ('train/4triples/WrittenWork.xml', 7832,
             '<sentence ID="1"/>'): False, (
                'train/4triples/WrittenWork.xml', 7833,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/4triples/WrittenWork.xml', 7837,
                '<sentence ID="3">'): '<sentence ID="2">',
            ('train/4triples/WrittenWork.xml', 7941,
             '<sentence ID="1"/>'): False, (
                'train/4triples/WrittenWork.xml', 7942,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('train/4triples/WrittenWork.xml', 10041,
             '<sentence ID="1"/>'): False, (
                'train/4triples/WrittenWork.xml', 10042,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/4triples/WrittenWork.xml', 10045,
                '<sentence ID="3">'): '<sentence ID="2">',
            ('train/4triples/WrittenWork.xml', 14017,
             '<sentence ID="1"/>'): False, (
                'train/4triples/WrittenWork.xml', 14018,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/4triples/WrittenWork.xml', 14151,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('train/4triples/WrittenWork.xml', 14150,
             '<sentence ID="1"/>'): False,
            ('train/5triples/Airport.xml', 17491, '<sentence ID="1"/>'): False,
            (
                'train/5triples/Airport.xml', 17492,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('train/5triples/Astronaut.xml', 710, '<sentence ID="1"/>'): False,
            (
                'train/5triples/Astronaut.xml', 711,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/5triples/Astronaut.xml', 1581,
                '<striple>Alan_Shepard | was selected by NASA | 1959</striple>'): False,
            ('train/5triples/Astronaut.xml', 1586,
             '<sentence ID="4"/>'): '<sentence ID="4"><striple>Alan_Shepard | was selected by NASA | 1959</striple></sentence>',
            ('train/5triples/Building.xml', 4249, '<sentence ID="1"/>'): False,
            (
                'train/5triples/Building.xml', 4250,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/5triples/Building.xml', 4255,
                '<sentence ID="3">'): '<sentence ID="2">', (
                'train/5triples/Building.xml', 4269,
                '<text>Founded in Washington.D.C. Marriot Hotels is the tenant of AC Hotel Bella Sky in Copenhagen Denmark. Denmark is led by Lars Lokke Rasmussen, where Faroese is spoken.</text>'): '<text>Founded in Washington.D.C., Marriot Hotels is the tenant of AC Hotel Bella Sky in Copenhagen Denmark. Denmark is led by Lars Lokke Rasmussen, where Faroese is spoken.</text>',
            ('train/5triples/Building.xml', 4270,
             '<template>Founded in PATIENT-2 . BRIDGE-2 is the tenant of AGENT-1 in BRIDGE-1 . BRIDGE-1 is led by PATIENT-3 , where PATIENT-1 is spoken .</template>'): '<template>Founded in PATIENT-2 , BRIDGE-2 is the tenant of AGENT-1 in BRIDGE-1 . BRIDGE-1 is led by PATIENT-3 , where PATIENT-1 is spoken .</template>',
            ('train/5triples/Building.xml', 4271,
             '<lexicalization>VP[aspect=simple,tense=past,voice=active,person=null,number=null] found in PATIENT-2 . BRIDGE-2 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=defined] the tenant of AGENT-1 in BRIDGE-1 . BRIDGE-1 VP[aspect=simple,tense=present,voice=passive,person=3rd,number=singular] lead by PATIENT-3 , where PATIENT-1 VP[aspect=simple,tense=present,voice=passive,person=3rd,number=singular] speak .</lexicalization>'): '<lexicalization>VP[aspect=simple,tense=past,voice=active,person=null,number=null] found in PATIENT-2 , BRIDGE-2 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=defined] the tenant of AGENT-1 in BRIDGE-1 . BRIDGE-1 VP[aspect=simple,tense=present,voice=passive,person=3rd,number=singular] lead by PATIENT-3 , where PATIENT-1 VP[aspect=simple,tense=present,voice=passive,person=3rd,number=singular] speak .</lexicalization>',
            ('train/5triples/Building.xml', 8042, '<sentence ID="4"/>'): False,
            (
                'train/5triples/Building.xml', 8057,
                '<text>&quot;Ampara Hospital is in Sri Lanka and is situated in the Eastern Province state of Sri Lanka. Austin Fernando is the leader of the Eastern Province of Sri Lanka and the Eastern Provincial Council is the governing body of Eastern Province, Sri Lanka. Sri Jayawardenepura Kotte is the capital of Sri Lanka.&quot;.</text>'): '<text>&quot;Ampara Hospital is in Sri Lanka and is situated in the Eastern Province state of Sri Lanka. Austin Fernando is the leader of the Eastern Province of Sri Lanka and the Eastern Provincial Council is the governing body of Eastern Province, Sri Lanka. Sri Jayawardenepura Kotte is the capital of Sri Lanka.&quot;</text>',
            ('train/5triples/Building.xml', 8058,
             '<template>`` AGENT-1 is in BRIDGE-2 and is situated in BRIDGE-1 state of BRIDGE-2 . PATIENT-2 is the leader of BRIDGE-1 of BRIDGE-2 and PATIENT-1 is the governing body of BRIDGE-1 . PATIENT-3 is the capital of BRIDGE-2 . '' .</template>'): '<template>`` AGENT-1 is in BRIDGE-2 and is situated in BRIDGE-1 state of BRIDGE-2 . PATIENT-2 is the leader of BRIDGE-1 of BRIDGE-2 and PATIENT-1 is the governing body of BRIDGE-1 . PATIENT-3 is the capital of BRIDGE-2 . ''</template>',
            ('train/5triples/Building.xml', 8059,
             '<lexicalization>`` AGENT-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be in BRIDGE-2 and VP[aspect=simple,tense=present,voice=passive,person=3rd,number=singular] situate in BRIDGE-1 state of BRIDGE-2 . PATIENT-2 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=defined] the leader of BRIDGE-1 of BRIDGE-2 and PATIENT-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=defined] the VP[aspect=progressive,tense=present,voice=active,person=null,number=null] govern body of BRIDGE-1 . PATIENT-3 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=defined] the capital of BRIDGE-2 . '' .</lexicalization>'): '<lexicalization>`` AGENT-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be in BRIDGE-2 and VP[aspect=simple,tense=present,voice=passive,person=3rd,number=singular] situate in BRIDGE-1 state of BRIDGE-2 . PATIENT-2 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=defined] the leader of BRIDGE-1 of BRIDGE-2 and PATIENT-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=defined] the VP[aspect=progressive,tense=present,voice=active,person=null, number=null] govern body of BRIDGE-1 . PATIENT-3 VP[aspect=simple,tense=present,voice=active,person=3rd, number=singular] be DT[form=defined] the capital of BRIDGE-2 . ''</lexicalization>',
            ('train/5triples/Food.xml', 12890, '</sentence>'): False,
            ('train/5triples/Food.xml', 12891, '<sentence ID="2">'): False, (
                'train/5triples/Food.xml', 12905,
                '<text>Coming from the region of Visayas, in the Philippines, Binignit, is a type of dessert. Which banana as the main ingredient but also has sago in it.</text>'): '<text>Coming from the region of Visayas, in the Philippines, Binignit, is a type of dessert, which banana as the main ingredient but also has sago in it.</text>',
            ('train/5triples/Food.xml', 12906,
             '<template>Coming from the region of PATIENT-1 , in PATIENT-4 , AGENT-1 , is a type of PATIENT-3 . Which PATIENT-2 as the main ingredient but also has PATIENT-5 in AGENT-1 .</template>'): '<template>Coming from the region of PATIENT-1 , in PATIENT-4 , AGENT-1 , is a type of PATIENT-3 , which PATIENT-2 as the main ingredient but also has PATIENT-5 in AGENT-1 .</template>',
            ('train/5triples/Food.xml', 12907,
             '<lexicalization>VP[aspect=progressive,tense=present,voice=active,person=null,number=null] come from DT[form=defined] the region of PATIENT-1 , in PATIENT-4 , AGENT-1 , VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=undefined] a type of PATIENT-3 . Which PATIENT-2 as DT[form=defined] the main ingredient but also VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have PATIENT-5 in AGENT-1 .</lexicalization>'): '<lexicalization>VP[aspect=progressive,tense=present,voice=active,person=null,number=null] come from DT[form=defined] the region of PATIENT-1 , in PATIENT-4 , AGENT-1 , VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=undefined] a type of PATIENT-3 , which PATIENT-2 as DT[form=defined] the main ingredient but also VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have PATIENT-5 in AGENT-1 .</lexicalization>',
            ('train/5triples/SportsTeam.xml', 70, '<sentence ID="1"/>'): False,
            (
                'train/5triples/SportsTeam.xml', 71,
                '<sentence ID="2">'): '<sentence ID="1">',
            (
                'train/5triples/SportsTeam.xml', 1304,
                '<sentence ID="1"/>'): False,
            (
                'train/5triples/SportsTeam.xml', 1305,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('train/5triples/SportsTeam.xml', 2913, '</sentence>'): False,
            ('train/5triples/SportsTeam.xml', 2914, '<sentence ID="2">'): False,
            (
                'train/5triples/SportsTeam.xml', 2929,
                "<text>Akron Summit Assault's ground is St. Vincent-St. Mary High School. Which is in the United States in Summit County, in Akron, Ohio where Dan Horrigan is the leader.</text>"): "<text>Akron Summit Assault's ground is St. Vincent-St. Mary High School, which is in the United States in Summit County, in Akron, Ohio where Dan Horrigan is the leader.</text>",
            ('train/5triples/SportsTeam.xml', 2930,
             '<template>AGENT-1 ground is BRIDGE-1 . Which is in PATIENT-3 in PATIENT-1 , in BRIDGE-2 where PATIENT-2 is the leader .</template>'): '<template>AGENT-1 ground is BRIDGE-1 , which is in PATIENT-3 in PATIENT-1 , in BRIDGE-2 where PATIENT-2 is the leader .</template>',
            ('train/5triples/SportsTeam.xml', 2931,
             '<lexicalization>AGENT-1 ground VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be BRIDGE-1 . Which VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be in PATIENT-3 in PATIENT-1 , in BRIDGE-2 where PATIENT-2 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=defined] the leader .</lexicalization>'): '<lexicalization>AGENT-1 ground VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be BRIDGE-1 , which VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be in PATIENT-3 in PATIENT-1 , in BRIDGE-2 where PATIENT-2 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=defined] the leader .</lexicalization>',
            (
                'train/5triples/SportsTeam.xml', 3262,
                '<sentence ID="1"/>'): False,
            (
                'train/5triples/SportsTeam.xml', 3263,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/5triples/SportsTeam.xml', 3267,
                '<sentence ID="3">'): '<sentence ID="2">', (
                'train/5triples/SportsTeam.xml', 3281,
                '<template>BRIDGE-1 is BRIDGE-1 . BRIDGE-1 which is located in the city of BRIDGE-2 (PATIENT-3) . PATIENT-1 is currently led by PATIENT-2 .</template>'): '<template>BRIDGE-1 is BRIDGE-1 which is located in the city of BRIDGE-2 (PATIENT-3) . PATIENT-1 is currently led by PATIENT-2 .</template>',
            ('train/5triples/SportsTeam.xml', 3282,
             '<lexicalization>BRIDGE-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be BRIDGE-1 . BRIDGE-1 which VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be located in DT[form=defined] the city of BRIDGE-2 ( PATIENT-3 ) . PATIENT-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be currently VP[aspect=simple,tense=past,voice=active,person=null,number=null] lead by PATIENT-2 .</lexicalization>'):
                '<lexicalization>BRIDGE-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be BRIDGE-1 which VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be located in DT[form=defined] the city of BRIDGE-2 ( PATIENT-3 ) . PATIENT-1 VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be currently VP[aspect=simple,tense=past,voice=active,person=null,number=null] lead by PATIENT-2 .</lexicalization>',
            (
                'train/5triples/University.xml', 1005,
                '<sentence ID="1"/>'): False,
            (
                'train/5triples/University.xml', 1006,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('train/5triples/WrittenWork.xml', 2241,
             '<sentence ID="1"/>'): False, (
                'train/5triples/WrittenWork.xml', 2242,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/5triples/WrittenWork.xml', 2246,
                '<sentence ID="3">'): '<sentence ID="2">',
            ('train/5triples/WrittenWork.xml', 2266,
             '<sentence ID="1"/>'): False, (
                'train/5triples/WrittenWork.xml', 2267,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/5triples/WrittenWork.xml', 2271,
                '<sentence ID="3">'): '<sentence ID="2">', (
                'train/5triples/WrittenWork.xml', 8181,
                '<sentence ID="1"/>'): '<sentence ID="1"><striple>United_States | leaderName | Barack_Obama</striple><striple>United_States | capital | Washington,_D.C.</striple></sentence>',
            ('train/5triples/WrittenWork.xml', 8182,
             '<sentence ID="2"/>'): '<sentence ID="2"><striple>1634:_The_Ram_Rebellion | country | United_States</striple><striple>United_States | ethnicGroup | African_Americans</striple></sentence>',
            ('train/6triples/Monument.xml', 2837, '<sentence ID="1"/>'): False,
            (
                'train/6triples/Monument.xml', 2838,
                '<sentence ID="2">'): '<sentence ID="1">',
            ('train/6triples/Monument.xml', 3632, '<sentence ID="1"/>'): False,
            (
                'train/6triples/Monument.xml', 3633,
                '<sentence ID="2">'): '<sentence ID="1">', (
                'train/7triples/Astronaut.xml', 10147,
                '<striple>Apollo_8 | operator | NASA</striple>'): False, (
                'train/7triples/Astronaut.xml', 10149,
                '<sentence ID="4"/>'): '<sentence ID="4"><striple>Apollo_8 | operator | NASA</striple></sentence>',
            ('train/7triples/University.xml', 31, '</sentence>'): False,
            ('train/7triples/University.xml', 32, '<sentence ID="2">'): False, (
                'train/7triples/University.xml', 46,
                '<text>The River Ganges flows through India which is the location of the AWH Engineering College which has 250 academic staff and was established in 2001 in the city of Kuttikkattoor in the state of Kerala. which is lead by Kochi.</text>'): '<text>The River Ganges flows through India which is the location of the AWH Engineering College which has 250 academic staff and was established in 2001 in the city of Kuttikkattoor in the state of Kerala, which is lead by Kochi.</text>',
            ('train/7triples/University.xml', 47,
             '<template>PATIENT-5 flows through BRIDGE-2 which is the location of AGENT-1 which has PATIENT-3 academic staff and was established in PATIENT-1 in the city of PATIENT-4 in the state of BRIDGE-1 . which is lead by PATIENT-2 .</template>'): '<template>PATIENT-5 flows through BRIDGE-2 which is the location of AGENT-1 which has PATIENT-3 academic staff and was established in PATIENT-1 in the city of PATIENT-4 in the state of BRIDGE-1 , which is lead by PATIENT-2 .</template>',
            ('train/7triples/University.xml', 48,
             '<lexicalization>PATIENT-5 VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] flow through BRIDGE-2 which VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=defined] the location of AGENT-1 which VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have PATIENT-3 academic staff and VP[aspect=simple,tense=past,voice=passive,person=null,number=singular] establish in PATIENT-1 in DT[form=defined] the city of PATIENT-4 in DT[form=defined] the state of BRIDGE-1 . which VP[aspect=simple,tense=present,voice=passive,person=3rd,number=singular] lead by PATIENT-2 .</lexicalization>'): '<lexicalization>PATIENT-5 VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] flow through BRIDGE-2 which VP[aspect=simple,tense=present,voice=active,person=3rd,number=singular] be DT[form=defined] the location of AGENT-1 which VP[aspect=simple,tense=present,voice=active,person=3rd,number=null] have PATIENT-3 academic staff and VP[aspect=simple,tense=past,voice=passive,person=null,number=singular] establish in PATIENT-1 in DT[form=defined] the city of PATIENT-4 in DT[form=defined] the state of BRIDGE-1 , which VP[aspect=simple,tense=present,voice=passive,person=3rd,number=singular] lead by PATIENT-2 .</lexicalization>',
            ('train/7triples/University.xml', 1261,
             '<sentence ID="1"/>'): '<sentence ID="1">', (
                'train/7triples/University.xml', 1262,
                '<sentence ID="2">'): '<striple>Switzerland | leaderName | Johann_Schneider-Ammann</striple>',
            ('train/7triples/University.xml', 1264,
             '<striple>Switzerland | leaderName | Johann_Schneider-Ammann</striple>'): '</sentence><sentence ID="2">',

        }


class NLP:
    def __init__(self):

        self.nlp = spacy.load('en', disable=['ner', 'parser', 'tagger'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def sent_tokenize(self, text):
        doc = self.nlp(text)
        sentences = [sent.string.strip() for sent in doc.sents]
        return sentences

    def word_tokenize(self, text, lower=False):  # create a tokenizer function
        if text is None: return text
        text = ' '.join(text.split())
        if lower: text = text.lower()
        toks = [tok.text for tok in self.nlp.tokenizer(text)]
        return ' '.join(toks)


def show_var(expression,
             joiner='\n', print=print):
    '''
    Prints out the name and value of variables.
    Eg. if a variable with name `num` and value `1`,
    it will print "num: 1\n"

    Parameters
    ----------
    expression: ``List[str]``, required
        A list of varible names string.

    Returns
    ----------
        None
    '''

    var_output = []

    for var_str in expression:
        frame = sys._getframe(1)
        value = eval(var_str, frame.f_globals, frame.f_locals)

        if ' object at ' in repr(value):
            value = vars(value)
            value = json.dumps(value, indent=2)
            var_output += ['{}: {}'.format(var_str, value)]
        else:
            var_output += ['{}: {}'.format(var_str, repr(value))]

    if joiner != '\n':
        output = "[Info] {}".format(joiner.join(var_output))
    else:
        output = joiner.join(var_output)
    print(output)
    return output


def fwrite(new_doc, path, mode='w', no_overwrite=False):
    if not path:
        print("[Info] Path does not exist in fwrite():", str(path))
        return
    if no_overwrite and os.path.isfile(path):
        print("[Error] pls choose whether to continue, as file already exists:",
              path)
        import pdb
        pdb.set_trace()
        return
    with open(path, mode) as f:
        f.write(new_doc)


def shell(cmd, working_directory='.', stdout=False, stderr=False):
    import subprocess
    from subprocess import PIPE, Popen

    subp = Popen(cmd, shell=True, stdout=PIPE,
                 stderr=subprocess.STDOUT, cwd=working_directory)
    subp_stdout, subp_stderr = subp.communicate()

    if stdout and subp_stdout:
        print("[stdout]", subp_stdout, "[end]")
    if stderr and subp_stderr:
        print("[stderr]", subp_stderr, "[end]")

    return subp_stdout, subp_stderr


def flatten_list(nested_list):
    from itertools import chain
    return list(chain.from_iterable(nested_list))


misspelling = {
    "accademiz": "academia",
    "withreference": "with reference",
    "thememorial": "the memorial",
    "unreleated": "unrelated",
    "varation": "variation",
    "variatons": "variations",
    "youthclub": "youth club",
    "oprated": "operated",
    "originaly": "originally",
    "origintes": "originates",
    "poacea": "poaceae",
    "posgraduayed": "postgraduate",
    "prevously": "previously",
    "publshed": "published",
    "punlished": "published",
    "recor": "record",
    "relgiion": "religion",
    "runwiay": "runway",
    "sequeled": "runway",
    "sppoken": "spoken",
    "studiies": "studies",
    "sytle": "style",
    "tboh": "both",
    "whic": "which",
    "identfier": "identifier",
    "idenitifier": "identifier",
    "igredient": "ingredients",
    "ingridient": "ingredients",
    "inclusdes": "includes",
    "indain": "indian",
    "leaderr": "leader",
    "legue": "league",
    "lenght": "length",
    "loaction": "location",
    "locaated": "located",
    "locatedd": "located",
    "locationa": "location",
    "managerof": "manager of",
    "manhattern": "manhattan",
    "memberrs": "members",
    "menbers": "members",
    "meteres": "metres",
    "numbere": "number",
    "numberr": "number",
    "notablework": "notable work",
    "7and": "7 and",
    "abbreivated": "abbreviated",
    "abreviated": "abbreviated",
    "abreviation": "abbreviation",
    "addres": "address",
    "abbreviatedform": "abbreviated form",
    "aerbaijan": "azerbaijan",
    "azerbijan": "azerbaijan",
    "affilaited": "affiliated",
    "affliate": "affiliate",
    "aircfrafts": "aircraft",
    "aircrafts": "aircraft",
    "aircarft": "aircraft",
    "airpor": "airport",
    "in augurated": "inaugurated",
    "inagurated": "inaugurated",
    "inaugrated": "inaugurated",
    "ausitin": "austin",
    "coccer": "soccer",
    "comanded": "commanded",
    "constructionof": "construction of",
    "counrty": "country",
    "countyof": "county of",
    "creater": "creator",
    "currecncy": "currency",
    "denonym": "demonym",
    "discipine": "discipline",
    "engish": "english",
    "establishedin": "established in",
    "ethinic": "ethnic",
    "ethiopa": "ethiopia",
    "ethipoia": "ethiopia",
    "eceived": "received",
    "ffiliated": "affiliated",
    "fullname": "full name",
    "grop": "group"
}

rephrasing = {
    # Add an acronym database
    "united states": ["u.s.", "u.s.a.", "us", "usa", "america", "american"],
    "united kingdom": ["u.k.", "uk"],
    "united states air force": ["usaf", "u.s.a.f"],
    "new york": ["ny", "n.y."],
    "new jersey": ["nj", "n.j."],
    "f.c.": ["fc"],
    "submarine": ["sub"],
    "world war ii": ["ww ii", "second world war"],
    "world war i": ["ww i", "first world war"],

    "greece": ["greek"],
    "canada": ["canadian"],
    "italy": ["italian"],
    "america": ["american"],
    "india": ["indian"],
    "singing": ["sings"],
    "conservative party (uk)": ["tories"],
    "ethiopia": ["ethiopian"],
}

rephrasing_must = {
    # Add a rephrasing database
    " language": "",
    " music": "",
    "kingdom of ": "",
    "new york city": "new york",
    "secretary of state of vermont": "secretary of vermont"
}


def rephrase(entity):
    phrasings = {entity}

    for s, rephs in rephrasing.items():
        for p in filter(lambda p: s in p, set(phrasings)):
            for r in rephs:
                phrasings.add(p.replace(s, r))

    # Allow rephrase "a/b/.../z" -> every permutation
    for p in set(phrasings):
        for permutation in itertools.permutations(p.split("/")):
            phrasings.add("/".join(permutation))

    # Allow rephrase "number (unit)" -> "number unit", "number unit-short"
    for p in set(phrasings):
        match = re.match("^(-?(\d+|\d{1,3}(,\d{3})*)(\.\d+)?)( (\((.*?)\)))?$",
                         p)
        if match:
            groups = match.groups()
            number = float(groups[0])
            unit = groups[6]

            number_phrasing = [
                str(number),
                str("{:,}".format(number))
            ]
            if round(number) == number:
                number_phrasing.append(str(round(number)))
                number_phrasing.append(str("{:,}".format(round(number))))

            if unit:
                couple = None
                words = [unit]

                if unit == "metres":
                    couple = "m"
                    words = [unit, "meters"]
                elif unit == "millimetres":
                    couple = "mm"
                elif unit == "centimetres":
                    couple = "cm"
                elif unit == "kilometres":
                    couple = "km"
                elif unit == "kilograms":
                    couple = "kg"
                elif unit == "litres":
                    couple = "l"
                elif unit == "inches":
                    couple = "''"
                elif unit in ["degreecelsius", "degreeklsius"]:
                    words = ["degrees celsius"]
                elif unit == "grampercubiccentimetres":
                    words = ["grams per cubic centimetre"]
                elif unit == "kilometreperseconds":
                    words = ["kilometres per second", "km/s", "km/sec",
                             "km per second", "km per sec"]
                elif unit in ["squarekilometres", "square kilometres"]:
                    words = ["square kilometres", "sq km"]
                elif unit == "cubiccentimetres":
                    couple = "cc"
                    words = ["cubic centimetres"]
                elif unit in ["cubic inches", "days", "tonnes", "square metres",
                              "inhabitants per square kilometre", "kelvins"]:
                    pass
                else:
                    raise ValueError(unit + " is unknown")

                for np in number_phrasing:
                    if couple:
                        phrasings.add(np + " " + couple)
                        phrasings.add(np + couple)
                    for word in words:
                        phrasings.add(np + " " + word)
            else:
                for np in number_phrasing:
                    phrasings.add(np)

    # Allow rephrase "word1 (word2)" -> "word2 word1"
    for p in set(phrasings):
        match = re.match("^(.* ?) \((.* ?)\)$", p)
        if match:
            groups = match.groups()
            s = groups[0]
            m = groups[1]
            phrasings.add(s + " " + m)
            phrasings.add(m + " " + s)

    return set(phrasings)


def rephrase_if_must(entity):
    phrasings = {entity}

    for s, rephs in rephrasing_must.items():
        for p in filter(lambda p: s in p, set(phrasings)):
            for r in rephs:
                phrasings.add(p.replace(s, r))

    # Allow removing parenthesis "word1 (word2)" -> "word1"
    for p in set(phrasings):
        match = re.match("^(.* ?) \((.* ?)\)$", p)
        if match:
            groups = match.groups()
            phrasings.add(groups[0])

    # Allow rephrase "word1 (word2) word3?" -> "word1( word3)"
    for p in set(phrasings):
        match = re.match("^(.*?) \((.*?)\)( .*)?$", p)
        if match:
            groups = match.groups()
            s = groups[0]
            m = groups[2]
            phrasings.add(s + " " + m if m else "")

    # Allow rephrase "a b ... z" -> every permutation
    # for p in set(phrasings):
    #     for permutation in itertools.permutations(p.split(" ")):
    #         phrasings.add(" ".join(permutation))

    phrasings = set(phrasings)
    if "" in phrasings:
        phrasings.remove("")
    return phrasings


fix_template_word = {
    '(AGENT-1': '( AGENT-1',
    '(AGENT-1)': '( AGENT-1',
    '(BRIDGE-1': '( BRIDGE-1',
    '(BRIDGE-1)': '( BRIDGE-1',
    '(BRIDGE-1)AGENT-1': '( BRIDGE-1 ) AGENT-1',
    '(BRIDGE-2': '( BRIDGE-2',
    '(BRIDGE-2)': '( BRIDGE-2',
    '(PATIENT-1': '( PATIENT-1',
    '(PATIENT-1)': '( PATIENT-1',
    '(PATIENT-1)AGENT-1': '( PATIENT-1 ) AGENT-1',
    '(PATIENT-1)PATIENT-2': '( PATIENT-1 ) PATIENT-2',
    '(PATIENT-1AGENT-1': '( PATIENT-1 AGENT-1',
    '(PATIENT-2': '( PATIENT-2',
    '(PATIENT-2)': '( PATIENT-2',
    '(PATIENT-2)AGENT-1': '( PATIENT-2 ) AGENT-1',
    '(PATIENT-2)PATIENT-1': '( PATIENT-2 ) PATIENT-1',
    '(PATIENT-2)PATIENT-3': '( PATIENT-2 ) PATIENT-3',
    '(PATIENT-2AGENT-1': '( PATIENT-2 AGENT-1',
    '(PATIENT-3': '( PATIENT-3',
    '(PATIENT-3)': '( PATIENT-3',
    '(PATIENT-3)AGENT-1': '( PATIENT-3 ) AGENT-1',
    '(PATIENT-3PATIENT-2': '( PATIENT-3 PATIENT-2',
    '(PATIENT-4)': '( PATIENT-4',
    '(PATIENT-5': '( PATIENT-5',
    '(PATIENT-5)': '( PATIENT-5',
    '.AGENT-1': '. AGENT-1',
    'groups.AGENT-1': 'groups . AGENT-1',
    'level.AGENT-1': 'level . AGENT-1',
    'number,PATIENT-2AGENT-1': 'number , PATIENT-2 AGENT-1',

    'AGENT-1,': 'AGENT-1 ,',
    'AGENT-1.': 'AGENT-1 .',
    'AGENT-1)': 'AGENT-1 )',
    'AGENT-1.AGENT-1': 'AGENT-1 . AGENT-1',
    'AGENT-1)AGENT-1': 'AGENT-1 ) AGENT-1',
    'AGENT-1AGENT-1': 'AGENT-1 AGENT-1',
    'AGENT-1AGENT-1PATIENT-1': 'AGENT-1 AGENT-1 PATIENT-1',
    'AGENT-1BRIDGE-1': 'AGENT-1 BRIDGE-1',
    'AGENT-1BRIDGE-2': 'AGENT-1 BRIDGE-2',
    'AGENT-1BRIDGE-3': 'AGENT-1 BRIDGE-3',
    'AGENT-1nAGENT-1': 'AGENT-1 AGENT-1',
    'AGENT-1PATIENT-1': 'AGENT-1 PATIENT-1',
    'AGENT-1PATIENT-2': 'AGENT-1 PATIENT-2',
    'AGENT-1PATIENT-3': 'AGENT-1 PATIENT-3',
    'AGENT-1PATIENT-4': 'AGENT-1 PATIENT-4',
    'AGENT-1PATIENT-5': 'AGENT-1 PATIENT-5',
    'BRIDGE-1': 'BRIDGE-1',
    'BRIDGE-1,': 'BRIDGE-1 ,',
    'BRIDGE-1.': 'BRIDGE-1 .',
    'BRIDGE-1)': 'BRIDGE-1 )',
    'BRIDGE-1.AGENT-1': 'BRIDGE-1 . AGENT-1',
    'BRIDGE-1)AGENT-1': 'BRIDGE-1 ) AGENT-1',
    'BRIDGE-1AGENT-1': 'BRIDGE-1 AGENT-1',
    'BRIDGE-1BRIDGE-1': 'BRIDGE-1 BRIDGE-1',
    'BRIDGE-1BRIDGE-2': 'BRIDGE-1 BRIDGE-2',
    'BRIDGE-1BRIDGE-3': 'BRIDGE-1 BRIDGE-3',
    'BRIDGE-1PATIENT-1': 'BRIDGE-1 PATIENT-1',
    'BRIDGE-1PATIENT-2': 'BRIDGE-1 PATIENT-2',
    'BRIDGE-1.PATIENT-3': 'BRIDGE-1 . PATIENT-3',
    'BRIDGE-1PATIENT-3': 'BRIDGE-1 PATIENT-3',
    'BRIDGE-1PATIENT-4': 'BRIDGE-1 PATIENT-4',
    'BRIDGE-1PATIENT-5': 'BRIDGE-1 PATIENT-5',
    'BRIDGE-2,': 'BRIDGE-2 ,',
    'BRIDGE-2.': 'BRIDGE-2 .',
    'BRIDGE-2)': 'BRIDGE-2 )',
    'BRIDGE-2.AGENT-1': 'BRIDGE-2 . AGENT-1',
    'BRIDGE-2AGENT-1': 'BRIDGE-2 AGENT-1',
    'BRIDGE-2BRIDGE-1': 'BRIDGE-2 BRIDGE-1',
    'BRIDGE-2BRIDGE-2': 'BRIDGE-2 BRIDGE-2',
    'BRIDGE-2BRIDGE-3': 'BRIDGE-2 BRIDGE-3',
    'BRIDGE-2PATIENT-1': 'BRIDGE-2 PATIENT-1',
    'BRIDGE-2PATIENT-2': 'BRIDGE-2 PATIENT-2',
    'BRIDGE-2PATIENT-3': 'BRIDGE-2 PATIENT-3',
    'BRIDGE-2.PATIENT-4': 'BRIDGE-2 . PATIENT-4',
    'BRIDGE-2PATIENT-4': 'BRIDGE-2 PATIENT-4',
    'BRIDGE-2PATIENT-6': 'BRIDGE-2 PATIENT-6',
    'BRIDGE-3': 'BRIDGE-3',
    'BRIDGE-3AGENT-1': 'BRIDGE-3 AGENT-1',
    'BRIDGE-3BRIDGE-1': 'BRIDGE-3 BRIDGE-1',
    'BRIDGE-3BRIDGE-2': 'BRIDGE-3 BRIDGE-2',
    'BRIDGE-3BRIDGE-3': 'BRIDGE-3 BRIDGE-3',
    'BRIDGE-3PATIENT-1': 'BRIDGE-3 PATIENT-1',
    'BRIDGE-3PATIENT-2': 'BRIDGE-3 PATIENT-2',
    'BRIDGE-3PATIENT-3': 'BRIDGE-3 PATIENT-3',
    'BRIDGE-4': 'BRIDGE-4',
    'BRIDGE-4AGENT-1': 'BRIDGE-4 AGENT-1',
    'BRIDGE-4BRIDGE-1': 'BRIDGE-4 BRIDGE-1',
    'BRIDGE-4PATIENT-2': 'BRIDGE-4 PATIENT-2',
    'PAGENT-1': 'AGENT-1',
    'PATIENT-1,': 'PATIENT-1 ,',
    'PATIENT-1.': 'PATIENT-1 .',
    'PATIENT-1)': 'PATIENT-1 )',
    'PATIENT-1))': 'PATIENT-1 ))',
    'PATIENT-1.AGENT-1': 'PATIENT-1 . AGENT-1',
    'PATIENT-1)AGENT-1': 'PATIENT-1 ) AGENT-1',
    'PATIENT-1AGENT-1': 'PATIENT-1 AGENT-1',
    'PATIENT-1AGENT-1AGENT-1': 'PATIENT-1 AGENT-1 AGENT-1',
    'PATIENT-1.BRIDGE-1': 'PATIENT-1 . BRIDGE-1',
    'PATIENT-1)BRIDGE-1': 'PATIENT-1 ) BRIDGE-1',
    'PATIENT-1BRIDGE-1': 'PATIENT-1 BRIDGE-1',
    'PATIENT-1)BRIDGE-2': 'PATIENT-1 ) BRIDGE-2',
    'PATIENT-1BRIDGE-2': 'PATIENT-1 BRIDGE-2',
    'PATIENT-1)BRIDGE-3': 'PATIENT-1 ) BRIDGE-3',
    'PATIENT-1BRIDGE-3': 'PATIENT-1 BRIDGE-3',
    'PATIENT-1PATIENT-1': 'PATIENT-1 PATIENT-1',
    'PATIENT-1PATIENT-1AGENT-1': 'PATIENT-1 PATIENT-1 AGENT-1',
    'PATIENT-1PATIENT-1PATIENT-1': 'PATIENT-1 PATIENT-1 PATIENT-1',
    'PATIENT-1PATIENT-2': 'PATIENT-1 PATIENT-2',
    'PATIENT-1(PATIENT-3)': 'PATIENT-1 ( PATIENT-3 )',
    'PATIENT-1)PATIENT-3': 'PATIENT-1 ) PATIENT-3',
    'PATIENT-1PATIENT-3': 'PATIENT-1 PATIENT-3',
    'PATIENT-1PATIENT-4': 'PATIENT-1 PATIENT-4',
    'PATIENT-1PATIENT-5': 'PATIENT-1 PATIENT-5',
    'PATIENT-1PATIENT-6': 'PATIENT-1 PATIENT-6',
    'PATIENT-2,': 'PATIENT-2 ,',
    'PATIENT-2.': 'PATIENT-2 .',
    'PATIENT-2)': 'PATIENT-2 )',
    'PATIENT-2.AGENT-1': 'PATIENT-2 . AGENT-1',
    'PATIENT-2)AGENT-1': 'PATIENT-2 ) AGENT-1',
    'PATIENT-2AGENT-1': 'PATIENT-2 AGENT-1',
    'PATIENT-2.BRIDGE-1': 'PATIENT-2 . BRIDGE-1',
    'PATIENT-2)BRIDGE-1': 'PATIENT-2 ) BRIDGE-1',
    'PATIENT-2BRIDGE-1': 'PATIENT-2 BRIDGE-1',
    'PATIENT-2.BRIDGE-2': 'PATIENT-2 . BRIDGE-2',
    'PATIENT-2BRIDGE-2': 'PATIENT-2 BRIDGE-2',
    'PATIENT-2BRIDGE-3': 'PATIENT-2 BRIDGE-3',
    'PATIENT-2CORRECT:AGENT-1': 'PATIENT-2 CORRECT: AGENT-1',
    'PATIENT-2)PATIENT-1': 'PATIENT-2 ) PATIENT-1',
    'PATIENT-2PATIENT-1': 'PATIENT-2 PATIENT-1',
    'PATIENT-2.PATIENT-2': 'PATIENT-2 . PATIENT-2',
    'PATIENT-2)PATIENT-2': 'PATIENT-2 ) PATIENT-2',
    'PATIENT-2PATIENT-2': 'PATIENT-2 PATIENT-2',
    'PATIENT-2)PATIENT-3': 'PATIENT-2 ) PATIENT-3',
    'PATIENT-2PATIENT-3': 'PATIENT-2 PATIENT-3',
    'PATIENT-2PATIENT-4': 'PATIENT-2 PATIENT-4',
    'PATIENT-2PATIENT-5': 'PATIENT-2 PATIENT-5',
    'PATIENT-2PATIENT-6': 'PATIENT-2 PATIENT-6',
    'PATIENT-2PATIENT-7': 'PATIENT-2 PATIENT-7',
    "PATIENT-2)'s": "PATIENT-2 ) 's",
    'PATIENT-3,': 'PATIENT-3 ,',
    'PATIENT-3.': 'PATIENT-3 .',
    'PATIENT-3)': 'PATIENT-3 )',
    'PATIENT-3.AGENT-1': 'PATIENT-3 . AGENT-1',
    'PATIENT-3)AGENT-1': 'PATIENT-3 )AGENT-1',
    'PATIENT-3AGENT-1': 'PATIENT-3 AGENT-1',
    'PATIENT-3.BRIDGE-1': 'PATIENT-3 . BRIDGE-1',
    'PATIENT-3)BRIDGE-1': 'PATIENT-3 ) BRIDGE-1',
    'PATIENT-3BRIDGE-1': 'PATIENT-3 BRIDGE-1',
    'PATIENT-3.BRIDGE-2': 'PATIENT-3 . BRIDGE-2',
    'PATIENT-3)BRIDGE-2': 'PATIENT-3 ) BRIDGE-2',
    'PATIENT-3BRIDGE-2': 'PATIENT-3 BRIDGE-2',
    'PATIENT-3BRIDGE-3': 'PATIENT-3 BRIDGE-3',
    'PATIENT-3CORRECT:AGENT-1': 'PATIENT-3 CORRECT: AGENT-1',
    'PATIENT-3)PATIENT-1': 'PATIENT-3 ) PATIENT-1',
    'PATIENT-3PATIENT-1': 'PATIENT-3 PATIENT-1',
    'PATIENT-3)PATIENT-2': 'PATIENT-3 ) PATIENT-2',
    'PATIENT-3PATIENT-2': 'PATIENT-3 PATIENT-2',
    'PATIENT-3)PATIENT-3': 'PATIENT-3 ) PATIENT-3',
    'PATIENT-3PATIENT-3': 'PATIENT-3 PATIENT-3',
    'PATIENT-3PATIENT-3AGENT-1': 'PATIENT-3 PATIENT-3 AGENT-1',
    'PATIENT-3PATIENT-4': 'PATIENT-3 PATIENT-4',
    'PATIENT-3.PATIENT-5': 'PATIENT-3 . PATIENT-5',
    'PATIENT-3PATIENT-5': 'PATIENT-3 PATIENT-5',
    'PATIENT-3PATIENT-6': 'PATIENT-3 PATIENT-6',
    'PATIENT-3PATIENT-7': 'PATIENT-3 PATIENT-7',
    'PATIENT-4,': 'PATIENT-4 ,',
    'PATIENT-4.': 'PATIENT-4 .',
    'PATIENT-4)': 'PATIENT-4 )',
    'PATIENT-4.AGENT-1': 'PATIENT-4 . AGENT-1',
    'PATIENT-4)AGENT-1': 'PATIENT-4 ) AGENT-1',
    'PATIENT-4AGENT-1': 'PATIENT-4 AGENT-1',
    'PATIENT-4BRIDGE-1': 'PATIENT-4 BRIDGE-1',
    'PATIENT-4BRIDGE-2': 'PATIENT-4 BRIDGE-2',
    '(PATIENT-4)PATIENT-1': '( PATIENT-4 ) PATIENT-1',
    'PATIENT-4.PATIENT-1': 'PATIENT-4 . PATIENT-1',
    'PATIENT-4PATIENT-1': 'PATIENT-4 PATIENT-1',
    'PATIENT-4.PATIENT-2': 'PATIENT-4 . PATIENT-2',
    'PATIENT-4PATIENT-2': 'PATIENT-4 PATIENT-2',
    'PATIENT-4PATIENT-3': 'PATIENT-4 PATIENT-3',
    'PATIENT-4(PATIENT-4)': 'PATIENT-4 ( PATIENT-4 )',
    'PATIENT-4PATIENT-4': 'PATIENT-4 PATIENT-4',
    'PATIENT-4PATIENT-5': 'PATIENT-4 PATIENT-5',
    'PATIENT-4PATIENT-6': 'PATIENT-4 PATIENT-6',
    'PATIENT-4PATIENT-7': 'PATIENT-4 PATIENT-7',
    'PATIENT-5,': 'PATIENT-5 ,',
    'PATIENT-5.': 'PATIENT-5 .',
    'PATIENT-5)': 'PATIENT-5 )',
    'PATIENT-5.AGENT-1': 'PATIENT-5 . AGENT-1',
    'PATIENT-5AGENT-1': 'PATIENT-5 AGENT-1',
    'PATIENT-5.BRIDGE-1': 'PATIENT-5 . BRIDGE-1',
    'PATIENT-5BRIDGE-1': 'PATIENT-5 BRIDGE-1',
    'PATIENT-5BRIDGE-2': 'PATIENT-5 BRIDGE-2',
    'PATIENT-5BRIDGE-3': 'PATIENT-5 BRIDGE-3',
    'PATIENT-5PATIENT-1': 'PATIENT-5 PATIENT-1',
    'PATIENT-5PATIENT-2': 'PATIENT-5 PATIENT-2',
    'PATIENT-5PATIENT-3': 'PATIENT-5 PATIENT-3',
    'PATIENT-5PATIENT-4': 'PATIENT-5 PATIENT-4',
    'PATIENT-5PATIENT-5': 'PATIENT-5 PATIENT-5',
    'PATIENT-5PATIENT-6': 'PATIENT-5 PATIENT-6',
    'PATIENT-6.': 'PATIENT-6 .',
    'PATIENT-6)': 'PATIENT-6 )',
    'PATIENT-6.AGENT-1': 'PATIENT-6 . AGENT-1',
    'PATIENT-6AGENT-1': 'PATIENT-6 AGENT-1',
    'PATIENT-6BRIDGE-1': 'PATIENT-6 BRIDGE-1',
    'PATIENT-6PATIENT-1': 'PATIENT-6 PATIENT-1',
    'PATIENT-6PATIENT-2': 'PATIENT-6 PATIENT-2',
    'PATIENT-6PATIENT-3': 'PATIENT-6 PATIENT-3',
    'PATIENT-6PATIENT-5': 'PATIENT-6 PATIENT-5',
    'PATIENT-6PATIENT-6': 'PATIENT-6 PATIENT-6',
    'PATIENT-6PATIENT-7': 'PATIENT-6 PATIENT-7',
    'PATIENT-7': 'PATIENT-7',
    'PATIENT-7,': 'PATIENT-7 ,',
    'PATIENT-7.': 'PATIENT-7 .',
    'PATIENT-7.AGENT-1': 'PATIENT-7 . AGENT-1',
    'PATIENT-7AGENT-1': 'PATIENT-7 AGENT-1',
    'PATIENT-7PATIENT-1': 'PATIENT-7 PATIENT-1',
    'PATIENT-7PATIENT-2': 'PATIENT-7 PATIENT-2',
    'PATIENT-7PATIENT-3': 'PATIENT-7 PATIENT-3',
    'PATIENT-7PATIENT-6': 'PATIENT-7 PATIENT-6',

}


def fix_tokenize(sentences):
    if sentences == [
        'The 11th Mississippi INfantry Monument is located in the municipality of Gettysburg, Adams County, Pa. It is categorized as a Contributing Property and was established in 2000.',
        'Carrol County, Maryland is southeast of Adams County, Pennsylvania.']:
        sentences = [
            'The 11th Mississippi INfantry Monument is located in the municipality of Gettysburg, Adams County, Pa.',
            'It is categorized as a Contributing Property and was established in 2000.',
            'Carrol County, Maryland is southeast of Adams County, Pennsylvania.']
    elif sentences == [
        'The 11th Mississipi Infantry Monument was established in 2000 and it is located in Adams County, Pa. The Monument is categorised as a contributing property.',
        'To the north of Adams County is Cumberland County (Pa), to its west is Franklin County (Pa) and to its southeast is Carrol County (Maryland).']:
        sentences = [
            'The 11th Mississipi Infantry Monument was established in 2000 and it is located in Adams County, Pa.',
            'The Monument is categorised as a contributing property.',
            'To the north of Adams County is Cumberland County (Pa), to its west is Franklin County (Pa) and to its southeast is Carrol County (Maryland).']
    elif sentences == [
        'United States @ test pilot @ Alan Bean was born in Wheeler, Texas.',
        'In 1955, he graduated from UT Austin with a B.S. Chosen by NASA in 1963, he managed a total space time of 100305.0 minutes.']:
        sentences = [
            'United States @ test pilot @ Alan Bean was born in Wheeler, Texas.',
            'In 1955, he graduated from UT Austin with a B.S.',
            'Chosen by NASA in 1963, he managed a total space time of 100305.0 minutes.']
    elif sentences == [
        'Alan Bean was originally from Wheeler, Texas and graduated from UT Austin in 1955 with a B.S. He went on to work as a test pilot and became a crew member of the Apollo 12 mission before he @ retired .']:
        sentences = [
            'Alan Bean was originally from Wheeler, Texas and graduated from UT Austin in 1955 with a B.S.',
            'He went on to work as a test pilot and became a crew member of the Apollo 12 mission before he @ retired .']
    elif sentences == [
        'Alan Bean is originally from Wheeler in Texas and graduated from UT Austin in 1955 with a B.S. He then went on to become a test pilot and became a member of the Apollo 12 crew where he spent 100305.0 minutes in space.']:
        sentences = [
            'Alan Bean is originally from Wheeler in Texas and graduated from UT Austin in 1955 with a B.S.',
            'He then went on to become a test pilot and became a member of the Apollo 12 crew where he spent 100305.0 minutes in space.']

    elif sentences == [
        'The American @ Alan Bean was born on the 15th of March 1932 in Wheeler, Texas.',
        'He graduated in 1955 from UT Austin with a B.S. He worked as a test pilot.']:
        sentences = [
            'The American @ Alan Bean was born on the 15th of March 1932 in Wheeler, Texas.',
            'He graduated in 1955 from UT Austin with a B.S.',
            'He worked as a test pilot.']
    elif sentences == [
        'Alan Bean was born in Wheeler, Texas and graduated in 1955 from UT Austin with a B.S. He performed as a test pilot and was chosen by NASA in 1963 to be the crew of Apollo 12.']:
        sentences = [
            'Alan Bean was born in Wheeler, Texas and graduated in 1955 from UT Austin with a B.S.',
            'He performed as a test pilot and was chosen by NASA in 1963 to be the crew of Apollo 12.']
    elif sentences == [
        'Alan Bean (born on March 15, 1932) graduated from UT Austin in 1955 with a B.S. Alan Bean who was chosen by NASA in 1963 is an American born in Wheeler, Texas.']:
        sentences = [
            'Alan Bean (born on March 15, 1932) graduated from UT Austin in 1955 with a B.S.',
            'Alan Bean who was chosen by NASA in 1963 is an American born in Wheeler, Texas.']
    elif sentences == [
        'Alan Shepherd born in New Hampshire, United States, graduated from NWC in 1957 with a M.A. He served as a crew member on Apollo 12 and retired in 1974.']:
        sentences = [
            'Alan Shepherd born in New Hampshire, United States, graduated from NWC in 1957 with a M.A.',
            'He served as a crew member on Apollo 12 and retired in 1974.']
    elif sentences == [
        'Alan Shepard was born in New Hampshire and graduated from NWC in 1957 with a M.A. Alan went on to become a test pilot and joined NASA in 1959.',
        'He later died in California.']:
        sentences = [
            'Alan Shepard was born in New Hampshire and graduated from NWC in 1957 with a M.A.',
            'Alan went on to become a test pilot and joined NASA in 1959.',
            'He later died in California.']
    elif sentences == [
        'Buzz Aldrin was born on Jan 20th, 1930 and his full name is Edwin E. Aldrin Jr. He graduated from MIT in 1963 with a doctorate in Science.',
        'He was a fighter pilot and a crew member of Apollo 11.',
        'He is now retired.']:
        sentences = [
            'Buzz Aldrin was born on Jan 20th, 1930 and his full name is Edwin E. Aldrin Jr.',
            'He graduated from MIT in 1963 with a doctorate in Science.',
            'He was a fighter pilot and a crew member of Apollo 11.',
            'He is now retired.']
    elif sentences == [
        'American @ Buzz Aldrin was born in Glen Ridge, New Jersey on January 20th, 1930.',
        'In 1963 he graduated from MIT with a Sc.',
        'D then became a fighter pilot and later a member of the Apollo 11 crew.']:
        sentences = [
            'American @ Buzz Aldrin was born in Glen Ridge, New Jersey on January 20th, 1930.',
            'In 1963 he graduated from MIT with a Sc. D then became a fighter pilot and later a member of the Apollo 11 crew.']
    elif sentences == [
        'Edwin E. Aldrin Jr. (more commonly known as Buzz) was born on January 20th, 1930.',
        'He graduated in 1963 from MIT with a Sc.',
        'D. befor becoming a fighter pilot and later a crew member of Apollo 11.']:
        sentences = [
            'Edwin E. Aldrin Jr. (more commonly known as Buzz) was born on January 20th, 1930.',
            'He graduated in 1963 from MIT with a Sc. D. befor becoming a fighter pilot and later a crew member of Apollo 11.']
    elif sentences == [
        'Buzz Aldrin was an American, who was born in Glen Ridge, NJ and graduated from MIT, Sc.',
        'D. in 1963.',
        'He was a fighter pilor and a member of the Apollo 11 crew.']:
        sentences = [
            'Buzz Aldrin was an American, who was born in Glen Ridge, NJ and graduated from MIT, Sc. D. in 1963.',
            'He was a fighter pilor and a member of the Apollo 11 crew.']
    elif sentences == [
        'Buzz Aldrin was a United States national who was born in Glen Ridge, New Jersey.',
        'He graduated from MIT with a Sc.', 'D in 1963.',
        'He served as a fighter pilot and became a crew member on Apollo 11.']:
        sentences = [
            'Buzz Aldrin was a United States national who was born in Glen Ridge, New Jersey.',
            'He graduated from MIT with a Sc. D in 1963.',
            'He served as a fighter pilot and became a crew member on Apollo 11.']
    elif sentences == [
        'Buzz Aldrin who was originally from New Jersey graduated from MIT with a Sc.',
        'D in 1963.',
        'He then went on to join NASA in 1963 and became a member of the Apollo 11 crew.']:
        sentences = [
            'Buzz Aldrin who was originally from New Jersey graduated from MIT with a Sc. D in 1963.',
            'He then went on to join NASA in 1963 and became a member of the Apollo 11 crew.']
    elif sentences == [
        'Buzz Aldrin is originally from Glen Ridge , New Jersey and graduated from MIT with a Sc.',
        'D in 1963.',
        'Buzz then went on to join NASA in 1963 and became a crew member of Apollo 11.']:
        sentences = [
            'Buzz Aldrin is originally from Glen Ridge , New Jersey and graduated from MIT with a Sc. D in 1963.',
            'Buzz then went on to join NASA in 1963 and became a crew member of Apollo 11.']
    elif sentences == [
        'Buzz Aldrin was born on the 20th of January, 1930 in Glen Ridge, New Jersey.',
        'He graduated from Massachusetts Institute of Technology, Sc.',
        'D. in 1963 and was selected to work for NASA the same year.',
        'He served as a crew member on Apollo 11.']:
        sentences = [
            'Buzz Aldrin was born on the 20th of January, 1930 in Glen Ridge, New Jersey.',
            'He graduated from Massachusetts Institute of Technology, Sc. D. in 1963 and was selected to work for NASA the same year.',
            'He served as a crew member on Apollo 11.']
    elif sentences == ['Buzz Aldrin graduated from MIT with a Sc.',
                       'D in 1963.',
                       'He was a fighter pilot and crew member of Apollo 11, which was organized by NASA.',
                       'William Anders was a back up pilot on the Apollo 11 mission.']:
        sentences = ['Buzz Aldrin graduated from MIT with a Sc. D in 1963.',
                     'He was a fighter pilot and crew member of Apollo 11, which was organized by NASA.',
                     'William Anders was a back up pilot on the Apollo 11 mission.']
    elif sentences == ['Buzz Aldrin graduated from MIT with a Sc.',
                       'D in 1963 and went on to become a fighter pilot with NASA .',
                       'He also became a part of the Apollo 11 crew.']:
        sentences = [
            'Buzz Aldrin graduated from MIT with a Sc. D in 1963 and went on to become a fighter pilot with NASA .',
            'He also became a part of the Apollo 11 crew.']
    elif sentences == [
        'William Anders who was born on October 17th 1933 in Hong Kong graduated in 1962 with a M.S. William Anders went on to become a member of the Apollo 8 crew and retired in 1969.']:
        sentences = [
            'William Anders who was born on October 17th 1933 in Hong Kong graduated in 1962 with a M.S.',
            'William Anders went on to become a member of the Apollo 8 crew and retired in 1969.']
    elif sentences == [
        'Elliot See was originally from Dallas and graduated from the University of Texas at Austin.',
        'He worked as a test pilot and died in St.',
        'Louis on the 28th February 1966.']:
        sentences = [
            'Elliot See was originally from Dallas and graduated from the University of Texas at Austin.',
            'He worked as a test pilot and died in St. Louis on the 28th February 1966.']
    elif sentences == [
        'Elliot See graduated from the University of Texas at Austin who are competing in the Big 12 Conference.',
        'Elliot See died in St.', 'Louis on February 28, 1966.']:
        sentences = [
            'Elliot See graduated from the University of Texas at Austin who are competing in the Big 12 Conference.',
            'Elliot See died in St. Louis on February 28, 1966.']
    elif sentences == [
        'Elliot See attended the University of Texas at Austin.',
        'The university is affiliated with the University of Texas System and it competed in the Big 12 Conference in Austin.',
        'The president of the university was Gregory L. Fenves.',
        'Elliot See died in St Louis.', 'The leader of St.',
        'Louis was Francis G. Slay.']:
        sentences = [
            'Elliot See attended the University of Texas at Austin.',
            'The university is affiliated with the University of Texas System and it competed in the Big 12 Conference in Austin.',
            'The president of the university was Gregory L. Fenves.',
            'Elliot See died in St Louis.',
            'The leader of St. Louis was Francis G. Slay.']
    elif sentences == [
        'Elliot See is originally from Dallas and joined NASA in 1962 where he flew as a test pilot.',
        'Elliot See died in St.', 'Louis.']:
        sentences = [
            'Elliot See is originally from Dallas and joined NASA in 1962 where he flew as a test pilot.',
            'Elliot See died in St. Louis.']
    elif sentences == [
        'B.M.Reddy is the President of the Acharya Institute of Technology which was founded in 2000 in India.',
        'The institute is also strongly connected to the Visvesvaraya Technological University which is located in Belgaum.',
        'The exact location for the Acharya Institute of Technology is " In Soldevanahalli, Acharya Dr.',
        'Sarvapalli Radhakrishnan Road, Hessarghatta Main Road, Bangalore - 560090.']:
        sentences = [
            'B.M.Reddy is the President of the Acharya Institute of Technology which was founded in 2000 in India.',
            'The institute is also strongly connected to the Visvesvaraya Technological University which is located in Belgaum.',
            'The exact location for the Acharya Institute of Technology is In Soldevanahalli, Acharya Dr. Sarvapalli Radhakrishnan Road, Hessarghatta Main Road, Bangalore - 560090.']
    elif sentences == [
        '1 Decembrie 1918 University is in the country of Romania, the capital of which is Bucharest.',
        "Romania's leader (who has the title Prime Minister) is Klaus Iohannis.",
        'The national anthem of Romania is Deșteaptă-te, române!',
        'and the country has an ethnic group called Germans of Romania.']:
        sentences = [
            '1 Decembrie 1918 University is in the country of Romania, the capital of which is Bucharest.',
            "Romania's leader (who has the title Prime Minister) is Klaus Iohannis.",
            'The national anthem of Romania is Deșteaptă-te, române! and the country has an ethnic group called Germans of Romania.']
    elif sentences == [
        'The Acharya Institute of Technology was founded in 2000 in the country India and has 700 postgraduate students.',
        'The institute has connections with the Visvesvaraya Technological University which is located in Belgaum.',
        'The exact location for the Acharya Institute of Technology is " In Soldevanahalli, Acharya Dr.',
        'Sarvapalli Radhakrishnan Road, Hessarghatta Main Road, Bangalore - 560090.".']:
        sentences = [
            'The Acharya Institute of Technology was founded in 2000 in the country India and has 700 postgraduate students.',
            'The institute has connections with the Visvesvaraya Technological University which is located in Belgaum.',
            'The exact location for the Acharya Institute of Technology is " In Soldevanahalli, Acharya Dr. Sarvapalli Radhakrishnan Road, Hessarghatta Main Road, Bangalore - 560090.".']
    elif sentences == [
        'The city of Aarhus in Denmark is served by an airport callled Aarhus Airport operated by Aarhus Lufthavn A/S. Runway 10L/28R is the longest runway there at a length of 2702 and is 25m above sea level.']:
        sentences = [
            'The city of Aarhus in Denmark is served by an airport callled Aarhus Airport operated by Aarhus Lufthavn A/S.',
            'Runway 10L/28R is the longest runway there at a length of 2702 and is 25m above sea level.']
    elif sentences == [
        'Aarhus Airport, which serves the city of Aarhus in Denmark, has a runway length of 2,776 and is named 10R/28L. Aktieselskab operates the airport which is 25 metres above sea level.']:
        sentences = [
            'Aarhus Airport, which serves the city of Aarhus in Denmark, has a runway length of 2,776 and is named 10R/28L.',
            'Aktieselskab operates the airport which is 25 metres above sea level.']
    elif sentences == [
        'Aarhus Airport in Denmark is operated by Aarhus Lufthavn A/S. The airport lies 25 metres above sea level and has a runway named 10R/28L which is 2776.0 metres long.']:
        sentences = [
            'Aarhus Airport in Denmark is operated by Aarhus Lufthavn A/S.',
            'The airport lies 25 metres above sea level and has a runway named 10R/28L which is 2776.0 metres long.']
    elif sentences == [
        'Aarhus Airport is located in Aarhus, Denmark and is operated by Aarhus Lufthavn A/S. The airport lies 25 metres above sea level and has a runway named 10L/28R which is 2777 metres long.']:
        sentences = [
            'Aarhus Airport is located in Aarhus, Denmark and is operated by Aarhus Lufthavn A/S.',
            'The  airport lies 25 metres above sea level and has a runway named 10L/28R which is 2777 metres long.']
    elif sentences == [
        'Aarhus airport services the city of Aarhus, Denmark and operated by Aarhus Lufthaven A/S. The airport is 25 meters above sea level and the 10L/28R runway is 2777.0 in length.']:
        sentences = [
            'Aarhus airport services the city of Aarhus, Denmark and operated by Aarhus Lufthaven A/S.',
            'The airport is 25 meters above sea level and the 10L/28R runway is 2777.0 in length.']
    elif sentences == [
        'The runway length of Adolfo Suárez Madrid–Barajas Airport is 3,500 and has the name 14L/32R. It is located at 610 metres above sea level in Madrid and is operated by ENAIRE.']:
        sentences = [
            'The runway length of Adolfo Suárez Madrid–Barajas Airport is 3,500 and has the name 14L/32R.',
            'It is located at 610 metres above sea level in Madrid and is operated by ENAIRE.']
    elif sentences == [
        'Aarhus Airport is located in Aarhus, Denmark, and is operated by Aarhus Lufthavn A/S. The airport is 25 meters above sea level, measuring 2777.0 in length, dubbed 10R/28L.']:
        sentences = [
            'Aarhus Airport is located in Aarhus, Denmark, and is operated by Aarhus Lufthavn A/S.',
            'The airport is 25 meters above sea level, measuring 2777.0 in length, dubbed 10R/28L.']
    elif sentences == [
        'Abilene Regional airport has a runway length of 1121.0 and is named 17L/35R. The airport serves Abilene in Texas, has the ICAO location identifier of KABI and is 546 metres above sea level.']:
        sentences = [
            'Abilene Regional airport has a runway length of 1121.0 and is named 17L/35R.',
            'The airport serves Abilene in Texas, has the ICAO location identifier of KABI and is 546 metres above sea level.']
    elif sentences == [
        'Abilene, Texas is served by the Abilene regional airport which is 546 metres above sea level.',
        'The airport has the ICAO Location Identifier, KABI, as well as having the runway name 17R/35L. One of the runways is 1121.0 metres long.']:
        sentences = [
            'Abilene, Texas is served by the Abilene regional airport which is 546 metres above sea level.',
            'The airport has the ICAO Location Identifier, KABI, as well as having the runway name 17R/35L.',
            'One of the runways is 1121.0 metres long.']
    elif sentences == [
        'Adolfo Suárez Madrid–Barajas Airport can be found in Madrid, Paracuellos de Jarama, San Sebastián de los Reyes and Alcobendas.',
        'It is operated by the ENAIRE organization.',
        "The airports's runway name is 18L/36R and its length is 3500 m. It is 610 m above sea level."]:
        sentences = [
            'Adolfo Suárez Madrid–Barajas Airport can be found in Madrid, Paracuellos de Jarama, San Sebastián de los Reyes and Alcobendas.',
            'It is operated by the ENAIRE organization.',
            "The airports's runway name is 18L/36R and its length is 3500 m.",
            "It is 610 m above sea level."]
    elif sentences == [
        'Operated by the United States Air Force, Al Asad Airbase is located in Al Anbar Province, Iraq.',
        "The Airbase's runway name is 09L/27R. Its ICAO Location Identifier is ORAA and 3992.88 is the length of the runway."]:
        sentences = [
            'Operated by the United States Air Force, Al Asad Airbase is located in Al Anbar Province, Iraq.',
            "The Airbase's runway name is 09L/27R.",
            "Its ICAO Location Identifier is ORAA and 3992.88 is the length of the runway."]
    elif sentences == [
        'Alpena County Regional Airport is located in Maple Ridge Township, Alpena County, Michigan and serves Alpena, Michigan.',
        'Its runway length is 1,533 and is named is 1/19/. The airport is 210 metres above sea level.']:
        sentences = [
            'Alpena County Regional Airport is located in Maple Ridge Township, Alpena County, Michigan and serves Alpena, Michigan.',
            'Its runway length is 1,533 and is named is 1/19/.',
            'The airport is 210 metres above sea level.']
    elif sentences == [
        'Alpena County Regional Airport city serves Alpena, Michigan in Wilson Township in the U.S. The airport is 210 m above sea level and 1533 m long.']:
        sentences = [
            'Alpena County Regional Airport city serves Alpena, Michigan in Wilson Township in the U.S.',
            'The airport is 210 m above sea level and 1533 m long.']
    elif sentences == [
        "Alpena County Regional Airport, which serves the city of Alpena, is located in Wilson Township, Alpena County, Michigan in the U.S.A. It's runway is 2,744 metres long and the facility is 210 metres above sea level."]:
        sentences = [
            "Alpena County Regional Airport, which serves the city of Alpena, is located in Wilson Township, Alpena County, Michigan in the U.S.A.",
            "It's runway is 2,744 metres long and the facility is 210 metres above sea level."]
    elif sentences == [
        'Andrews County Airport is located in Texas, U.S. The inhabitants of Texas have the demonym of Tejano and Spanish is spoken.',
        'The capital is Austin.']:
        sentences = ['Andrews County Airport is located in Texas, U.S.',
                     'The inhabitants of Texas have the demonym of Tejano and Spanish is spoken.',
                     'The capital is Austin.']
    elif sentences == [
        'Texas maintains the capital as Austin and is the home of Houston (the largest city in TX.)',
        'and the Andrews County Airport.', 'Tejanos are people of Texas.']:
        sentences = [
            'Texas maintains the capital as Austin and is the home of Houston (the largest city in TX.) and the Andrews County Airport.',
            'Tejanos are people of Texas.']
    elif sentences == [
        'Andrews County Airport is located in Texas in the U.S. The capital of Texas is Austin and its largest city is Houston.',
        'English is spoken in that state.']:
        sentences = [
            'Andrews County Airport is located in Texas in the U.S.',
            'The capital of Texas is Austin and its largest city is Houston.',
            'English is spoken in that state.']
    elif sentences == [
        "Atlantic City International Airport in Egg Harbor Township, N.J. serves Atlantic City in the U.S.A. The city's leader is Don Guardian."]:
        sentences = [
            "Atlantic City International Airport in Egg Harbor Township, N.J. serves Atlantic City in the U.S.A.",
            "The city's leader is Don Guardian."]
    elif sentences == [
        'Atlantic City International Airport serves the city of Atlantic City, New Jersey in the U.S.A. The airport is in Egg Harbor Township, New Jersey.',
        'The Atlantic City, New Jersey leader is Don Guardian.']:
        sentences = [
            'Atlantic City International Airport serves the city of Atlantic City, New Jersey in the U.S.A.',
            'The airport is in Egg Harbor Township, New Jersey.',
            'The Atlantic City, New Jersey leader is Don Guardian.']
    elif sentences == [
        'Atlantic City International Airport in Egg Harbor Township, New Jersey is in the U.S.A. The airport has a runway that is 1,873 long.']:
        sentences = [
            'Atlantic City International Airport in Egg Harbor Township, New Jersey is in the U.S.A.',
            'The airport has a runway that is 1,873 long.']
    elif sentences == [
        'Bacon Explosion which has bacon and sausage in it comes from Kansas City metro area in the U.S. The Bacon Explosion is a main course.']:
        sentences = [
            'Bacon Explosion which has bacon and sausage in it comes from Kansas City metro area in the U.S.',
            'The Bacon Explosion is a main course.']
    elif sentences == [
        'The bacon explosion took place in the U.S.A. where Paul Ryan is leader.',
        "The country's capital is Washington, D.C. The president leads the U.S. and among its ethnic groups are white Americans."]:
        sentences = [
            'The bacon explosion took place in the U.S.A. where Paul Ryan is leader.',
            "The country's capital is Washington, D.C.",
            "The president leads the U.S. and among its ethnic groups are white Americans."]
    elif sentences == [
        'The Bacon Explosion comes from the United States, a country whose leader has the title of President and whose capital is Washington, D.C. One of the leaders of the U.S. is Barack Obama and one of the ethnic groups is the African Americans.']:
        sentences = [
            'The Bacon Explosion comes from the United States, a country whose leader has the title of President and whose capital is Washington, D.C.',
            'One of the leaders of the U.S. is Barack Obama and one of the ethnic groups is the African Americans.']
    elif sentences == [
        'Bacon Explosion comes from the United States where Asian Americans are an ethnic group and the capital is Washington, D.C. The leader of the United States is called the President and this is Barack Obama.']:
        sentences = [
            'Bacon Explosion comes from the United States where Asian Americans are an ethnic group and the capital is Washington, D.C.',
            'The leader of the United States is called the President and this is Barack Obama.']
    elif sentences == [
        'White Americans are one of the ethnic groups in the United States, a country where the the leader is called the President and Washington, D.C. is the capital city.',
        'Joe Biden is a political leader in the U.S. The country is also the origin of Bacon Explosion.']:
        sentences = [
            'White Americans are one of the ethnic groups in the United States, a country where the the leader is called the President and Washington, D.C. is the capital city.',
            'Joe Biden is a political leader in the U.S.',
            'The country is also the origin of Bacon Explosion.']
    elif sentences == [
        '200 Public Square is in Cleveland, Ohio (part of Cuyahoga County) in the U.S. It has 45 floors.']:
        sentences = [
            '200 Public Square is in Cleveland, Ohio (part of Cuyahoga County) in the U.S.',
            'It has 45 floors.']
    elif sentences == [
        '300 North LaSalle is in Chicago which is part of Cook County, Illinois in the U.S. The leader is Susana Mendoza.']:
        sentences = [
            '300 North LaSalle is in Chicago which is part of Cook County, Illinois in the U.S.',
            'The leader is Susana Mendoza.']
    elif sentences == [
        "300 North LaSalle, with 60 floors, is located in Chicago, Illinois, U.S.. Chicago's leader is called Rahm Emanuel."]:
        sentences = [
            "300 North LaSalle, with 60 floors, is located in Chicago, Illinois, U.S.",
            "Chicago's leader is called Rahm Emanuel."]
    elif sentences == [
        'The address of Amdavad ni Gufa is Lalbhai Dalpatbhai Campus, near CEPT University, opp.',
        'Gujarat University, University Road, Gujarat, Ahmedabad, India.',
        'Amdavad ni Gufa was completed in 1995.']:
        sentences = [
            'The address of Amdavad ni Gufa is Lalbhai Dalpatbhai Campus, near CEPT University, opp. Gujarat University, University Road, Gujarat, Ahmedabad, India.',
            'Amdavad ni Gufa was completed in 1995.']
    elif sentences == [
        'Asilomar Conference Grounds which was constructed in 1913 in the architectural style of American Craftsman is located at Asilomar Blvd.,',
        'Pacific Grove, California.',
        'It was added to the National Register of Historic Places on 27 February 1987 with the reference number 87000823.']:
        sentences = [
            'Asilomar Conference Grounds which was constructed in 1913 in the architectural style of American Craftsman is located at Asilomar Blvd., Pacific Grove, California.',
            'It was added to the National Register of Historic Places on 27 February 1987 with the reference number 87000823.']
    elif sentences == [
        'The location of Asilomar Conference Grounds which were constructed in 1913 is Asilomar Blvd.,',
        'Pacific Grove, California.',
        'They were added to the National Register of Historic Places on 27 February 1987 with the reference number "87000823", and were built in the Arts and Crafts Movement architectural style.']:
        sentences = [
            'The location of Asilomar Conference Grounds which were constructed in 1913 is Asilomar Blvd., Pacific Grove, California.',
            'They were added to the National Register of Historic Places on 27 February 1987 with the reference number "87000823", and were built in the Arts and Crafts Movement architectural style.']
    elif sentences == [
        'Asser Levy Public Baths are found in New York City, Manhattan, New York, in the U.S.. Cyrus Vance Jr. is one of the leaders of Manhattan.']:
        sentences = [
            'Asser Levy Public Baths are found in New York City, Manhattan, New York, in the U.S..',
            'Cyrus Vance Jr. is one of the leaders of Manhattan.']
    elif sentences == [
        'Baymax is a character in the film Big Hero 6 which stars Damon Wayans Jr. He was created by Steven t Seagle and the American, Duncan Rouleau.']:
        sentences = [
            'Baymax is a character in the film Big Hero 6 which stars Damon Wayans Jr.',
            'He was created by Steven t Seagle and the American, Duncan Rouleau.']
    elif sentences == [
        'Stuart Parker plays for KV Mechelen and the Blackburn Rovers F.C. AFC Blackpool(Blackpool) had Stuart Parker as their manager.',
        'The Conservative Party U.K. is the leader of Blackpool.']:
        sentences = [
            'Stuart Parker plays for KV Mechelen and the Blackburn Rovers F.C.',
            'AFC Blackpool(Blackpool) had Stuart Parker as their manager.',
            'The Conservative Party U.K. is the leader of Blackpool.']
    elif sentences == [
        'A.S. Gubbio 1910 (Italy) play in Serie D. S.S. Robur Siena are champions of that serie.',
        'Pietro Grasso leads Italy.',
        'Italian is the language spoken in Italy.']:
        sentences = ['A.S. Gubbio 1910 (Italy) play in Serie D. S.S.',
                     'Robur Siena are champions of that serie.',
                     'Pietro Grasso leads Italy.',
                     'Italian is the language spoken in Italy.']
    elif sentences == ['St. Vincent-St.',
                       'Mary High School, which is in Akron, Ohio, in the United States, is the ground of Akron Summit Assault.',
                       'They play in the Premier Development League, where the champions have been K-W United FC.']:
        sentences = [
            'St. Vincent-St. Mary High School, which is in Akron, Ohio, in the United States, is the ground of Akron Summit Assault.',
            'They play in the Premier Development League, where the champions have been K-W United FC.']
    elif sentences == [
        'K-W United FC have been champions of the Premier Development League, which Akron Summit Assault play in.',
        'their ground is St. Vincent-St.',
        'Mary High School in Akron, Ohio, in the U.S.']:
        sentences = [
            'K-W United FC have been champions of the Premier Development League, which Akron Summit Assault play in.',
            'their ground is St. Vincent-St. Mary High School in Akron, Ohio, in the U.S.']
    elif sentences == ['St. Vincent–St.',
                       'Mary High School is located in the city of Akron, Ohio, USA.',
                       'The school is the ground of Akron Summit Assault.',
                       'The city is part of Summit County, Ohio.',
                       'It is led by Dan Horrigan.']:
        sentences = [
            'St. Vincent–St. Mary High School is located in the city of Akron, Ohio, USA.',
            'The school is the ground of Akron Summit Assault.',
            'The city is part of Summit County, Ohio.',
            'It is led by Dan Horrigan.']
    elif sentences == ['St. Vincent-St.',
                       'Mary High School is located in Akron, Summit County, Ohio @ USA.',
                       'St Vincent-St Mary High School is the ground of Akron Summit Assault.',
                       'The leader of Akron, Ohio is a one Dan Horrigan.']:
        sentences = [
            'St. Vincent-St. Mary High School is located in Akron, Summit County, Ohio @ USA.',
            'St Vincent-St Mary High School is the ground of Akron Summit Assault.',
            'The leader of Akron, Ohio is a one Dan Horrigan.']
    elif sentences == ["Akron Summit Assault's ground is St. Vincent-St.",
                       'Mary High School.',
                       'Which is in the United States in Summit County, in Akron, Ohio where Dan Horrigan is the leader.']:
        sentences = [
            "Akron Summit Assault's ground is St. Vincent-St. Mary High School, Which is in the United States in Summit County, in Akron, Ohio where Dan Horrigan is the leader."]
    elif sentences == ['St. Vincent St.',
                       'Mary High School is located in Summit County, Ohio, Akron, Ohio, United States.',
                       'Its leader is Dan Horrigan and it is the ground of Akron Summit Assault.']:
        sentences = [
            'St. Vincent St. Mary High School is located in Summit County, Ohio, Akron, Ohio, United States.',
            'Its leader is Dan Horrigan and it is the ground of Akron Summit Assault.']
    elif sentences == [
        'K-W United FC have been champions of the Premier Development League, which is the league Akron Summit Assault play in.',
        "Akron Summit Assault's ground is St. Vincent-St.",
        'Mary High School, in Akron, Ohio, in the United States.']:
        sentences = [
            'K-W United FC have been champions of the Premier Development League, which is the league Akron Summit Assault play in.',
            "Akron Summit Assault's ground is St. Vincent-St. Mary High School, in Akron, Ohio, in the United States."]
    elif sentences == ["Akron Summit Assault's ground is St. Vincent-St.",
                       'Mary High School, Akron, Ohio, United States.',
                       'The team play in the Premier Development League, which has previously been won by K-W United FC.']:
        sentences = [
            "Akron Summit Assault's ground is St. Vincent-St. Mary High School, Akron, Ohio, United States.",
            'The team play in the Premier Development League, which has previously been won by K-W United FC.']
    elif sentences == ['St. Vincent-St.',
                       'Mary High School is in Akron, Ohio @ U.S and is the home ground for Akron Summit Assault.',
                       'Dan Horrigan is the leader of Akron, Ohio.']:
        sentences = [
            'St. Vincent-St. Mary High School is in Akron, Ohio @ U.S and is the home ground for Akron Summit Assault.',
            'Dan Horrigan is the leader of Akron, Ohio.']
    elif sentences == [
        "The Akron Summit Assault's ground is St. Vincent-St.",
        'Mary High School.',
        'The School is located in Akron, Ohio, United States which currently has Dan Horrigan as a leader.']:
        sentences = [
            "The Akron Summit Assault's ground is St. Vincent-St. Mary High School.', 'The School is located in Akron, Ohio, United States which currently has Dan Horrigan as a leader."]
    elif sentences == ['St. Vincent St.',
                       'Mary High School is located in Summit County, Ohio in the United States.',
                       'The Akron Summit Assault ground is at this high school.']:
        sentences = [
            'St. Vincent St. Mary High School is located in Summit County, Ohio in the United States.',
            'The Akron Summit Assault ground is at this high school.']
    elif sentences == [
        'The Olympic Stadium (in Athens) is the home ground of AEK Athens FC.',
        'That football team is managed by Gus Poyet who played for Chelsea F.C. Gus Poyet is also associated with the Real Zaragoza football club.']:
        sentences = [
            'The Olympic Stadium (in Athens) is the home ground of AEK Athens FC.',
            'That football team is managed by Gus Poyet who played for Chelsea F.C.',
            'Gus Poyet is also associated with the Real Zaragoza football club.']
    elif sentences == [
        'The Acharya Institute of Technology is located in the city of Bangalore in India and was established in 2000.',
        "The Institute's President is B.M. Reddy and the Director is Dr.",
        'G.P.Prabhukumar.']:
        sentences = [
            'The Acharya Institute of Technology is located in the city of Bangalore in India and was established in 2000.',
            "The Institute's President is B.M. Reddy and the Director is Dr. G.P.Prabhukumar."]
    elif sentences == [
        'Bangalore was founded by Kempe Gowda I. Located in the city is the Acharya Institute of Technology, an affiliate of Visvesvaraya Technological University.',
        'The institute offers tennis, as governed by the International Tennis Federation, as a sport.']:
        sentences = ['Bangalore was founded by Kempe Gowda I.',
                     'Located in the city is the Acharya Institute of Technology, an affiliate of Visvesvaraya Technological University.',
                     'The institute offers tennis, as governed by the International Tennis Federation, as a sport.']
    elif sentences == ['Acta Palaeontologica Polonica (abbr.',
                       'Acta Palaeontol.',
                       'Pol) is published by the Institute of Paleobiology, Polish Academy of Sciences.',
                       'Code information: ISSN number 0567-7920, LCCN number of 60040714, CODEN code APGPAC.']:
        sentences = [
            'Acta Palaeontologica Polonica (abbr. Acta Palaeontol Pol) is published by the Institute of Paleobiology, Polish Academy of Sciences.',
            'Code information: ISSN number 0567-7920, LCCN number of 60040714, CODEN code APGPAC.']
    elif sentences == [
        'English is spoken in Great Britain and Alcatraz Versus the Evil Librarians was written in it but comes from the U.S. Native Americans are one of the ethnic groups of the United States and the capital city is Washington D.C.']:
        sentences = [
            'English is spoken in Great Britain and Alcatraz Versus the Evil Librarians was written in it but comes from the U.S.',
            'Native Americans are one of the ethnic groups of the United States and the capital city is Washington D.C.']
    elif sentences == [
        'A Loyal Character Dancer is published by Soho Press in the U.S. The language spoken there is English, which was originated in Great Britain.',
        'One ethnic group of the U.S. is African American.']:
        sentences = [
            'A Loyal Character Dancer is published by Soho Press in the U.S.',
            'The language spoken there is English, which was originated in Great Britain.',
            'One ethnic group of the U.S. is African American.']
    elif sentences == [
        '1634 The Ram Rebellion (preceded by 1634: The Galileo Affair) comes from the United States, where Barack Obama is the President, and its capital city is Washington D.C. Native Americans are one of the ethnic groups of the United States.']:
        sentences = [
            '1634 The Ram Rebellion (preceded by 1634: The Galileo Affair) comes from the United States, where Barack Obama is the President, and its capital city is Washington D.C.',
            'Native Americans are one of the ethnic groups of the United States.']
    return sentences


if __name__ == '__main__':
    cl = Cleaner(verbose=True)
