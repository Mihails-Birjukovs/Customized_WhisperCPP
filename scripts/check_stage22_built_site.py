"""Generated HTML/CSS smoke validation; not live browser visual acceptance."""
from pathlib import Path
from html.parser import HTMLParser
import json
ROOT=Path(__file__).resolve().parents[1]
SITE=ROOT/"site"
class SiteLinks(HTMLParser):
 def __init__(self):
  super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  if tag=="link":
   a=dict(attrs)
   if a.get("rel")=="stylesheet" or "stylesheet" in (a.get("rel") or "").split():
    self.links.append(a.get("href",""))
def verify():
 html=SITE/"index.html"
 css=SITE/"stylesheets/inm-documentation-v1.css"
 if not html.is_file() or not css.is_file():
  raise ValueError("Built HTML/CSS missing")
 page=html.read_text(encoding="utf-8")
 if "Recover from output loops" not in page or "27/27" not in page:
  raise ValueError("Hero/evidence not rendered in HTML")
 for x in ("--inm-blue","#004b8e","--inm-orange","#f47b20"):
  if x not in css.read_text(encoding="utf-8"):raise ValueError("Design token missing")
 parser=SiteLinks();parser.feed(page)
 if not any("inm-documentation-v1.css" in link for link in parser.links):
  raise ValueError("Built page does not load INM stylesheet")
 for slug in ("getting-started","architecture","adaptive-recovery","validation",
              "provenance","roadmap","credits"):
  if not (SITE/slug/"index.html").is_file():
   raise ValueError("Missing navigation target: "+slug)
 if "mathjax@3.2.2" not in page or "javascripts/mathjax.js" not in page or not (SITE/"javascripts/mathjax.js").is_file():
  raise ValueError("Generated site lacks pinned MathJax runtime and script")
 for name in ("inm-horizontal.svg","inm-mark.svg","inm-square-footer.svg"):
  if not (SITE/"assets"/name).is_file():raise ValueError("Official reviewed INM artwork missing from generated site: "+name)
 if "assets/inm-horizontal.svg" not in page or "assets/inm-mark.svg" not in page:
  raise ValueError("Header INM logo or favicon absent from generated page")
 if "inm-square-footer.svg" not in css.read_text(encoding="utf-8"):
  raise ValueError("Approved INM footer logo CSS absent")
 if "md-source" not in page or "github.com/Mihails-Birjukovs/Customized_WhisperCPP" not in page:
  raise ValueError("Public GitHub repository icon link missing from header")
 return {"html_and_css_present":True,"nav_pages":8,
         "private_data_review_not_implied":True,"site_deployed":False}
if __name__=="__main__":
 print(json.dumps(verify(),indent=2))
