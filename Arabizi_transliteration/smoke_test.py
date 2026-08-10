import sys
sys.stdout.reconfigure(encoding='utf-8')
from transliterate import transliterate

inputs = ['راني عارف', 'عليهم', 'وشنو راك دير', 'ربي يحفظك خويا', 'الدار في وهران']
for t in inputs:
    print(f"{t}  ->  {transliterate(t)}")
