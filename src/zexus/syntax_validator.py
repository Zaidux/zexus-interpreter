# syntax_validator.py
import re

class SyntaxValidator:
    def __init__(self):
        self.suggestions = []
        self.warnings = []

    def validate_code(self, code, desired_style="universal"):
        """Validate code and suggest improvements for the desired syntax style"""
        self.suggestions = []
        self.warnings = []

        lines = code.split('\n')

        for i, line in enumerate(lines):
            line_num = i + 1
            self._validate_line(line, line_num, desired_style)

        return {
            'is_valid': len(self.suggestions) == 0,
            'suggestions': self.suggestions,
            'warnings': self.warnings,
            'error_count': len(self.suggestions)
        }

    def _validate_line(self, line, line_num, style):
        """Validate a single line against the desired style"""
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('#'):
            return

        # Universal style validations
        if style == "universal":
            self._validate_universal_syntax(stripped_line, line_num, line)
        # Tolerable style validations  
        elif style == "tolerable":
            self._validate_tolerable_syntax(stripped_line, line_num, line)

        # Common validations for both styles
        self._validate_common_syntax(stripped_line, line_num, line)

    def _validate_universal_syntax(self, stripped_line, line_num, original_line):
        """Validate against universal syntax rules"""
        # Check for colon blocks (should use braces)
        if (any(stripped_line.startswith(keyword) for keyword in ['if', 'for each', 'while', 'action', 'try']) 
            and stripped_line.endswith(':')):
            self.suggestions.append({
                'line': line_num,
                'message': "Universal syntax requires braces {} instead of colon for blocks",
                'fix': original_line.rstrip(':') + " {",
                'severity': 'warning'
            })

        # Check debug statements
        if stripped_line.startswith('debug ') and not stripped_line.startswith('debug('):
            self.suggestions.append({
                'line': line_num,
                'message': "Use parentheses with debug statements: debug(expression)",
                'fix': original_line.replace('debug ', 'debug(', 1) + ')',
                'severity': 'error'
            })

        # Check lambda syntax
        if 'lambda' in stripped_line and 'lambda ' in stripped_line and not 'lambda(' in stripped_line:
            self.suggestions.append({
                'line': line_num,
                'message': "Use parentheses with lambda parameters: lambda(params) -> expression",
                'fix': self._fix_lambda_syntax(original_line),
                'severity': 'error'
            })

        # Check catch without parentheses or with double parentheses
        if 'catch' in stripped_line:
            # Case 1: catch without parentheses: catch error { }
            # Check for catch followed by space, not followed by ( (with or without space before it)
            if ('catch ' in stripped_line and 
                not 'catch(' in stripped_line and 
                not 'catch (' in stripped_line and 
                '{' in stripped_line):
                self.suggestions.append({
                    'line': line_num,
                    'message': "Use parentheses with catch: catch(error) { }",
                    'fix': self._fix_catch_syntax(original_line),
                    'severity': 'error'
                })
            # Case 2: catch with double parentheses: catch((error)) { }
            elif 'catch((' in stripped_line and '))' in stripped_line:
                self.suggestions.append({
                    'line': line_num,
                    'message': "Remove extra parentheses in catch: catch(error) { }",
                    'fix': self._fix_double_parentheses_catch(original_line),
                    'severity': 'error'
                })

    def _validate_tolerable_syntax(self, stripped_line, line_num, original_line):
        """Validate against tolerable syntax rules"""
        # Check for potentially confusing syntax
        if stripped_line.count('{') != stripped_line.count('}'):
            self.warnings.append({
                'line': line_num,
                'message': "Mismatched braces - this can cause parsing issues",
                'severity': 'warning'
            })

        # Check for mixed block styles in same context
        if (any(stripped_line.startswith(keyword) for keyword in ['if', 'for each', 'while']) 
            and ':' in stripped_line and '{' in stripped_line):
            self.suggestions.append({
                'line': line_num,
                'message': "Mixed block syntax - prefer consistent use of : or {}",
                'fix': original_line,
                'severity': 'warning'
            })

    def _validate_common_syntax(self, stripped_line, line_num, original_line):
        """Common validations for both syntax styles"""
        # Check for missing parentheses in function calls
        if (any(stripped_line.startswith(keyword) for keyword in ['if', 'while']) 
            and '(' not in stripped_line and not stripped_line.endswith(':')):
            self.suggestions.append({
                'line': line_num,
                'message': "Consider using parentheses for clarity: if (condition)",
                'fix': self._add_parentheses_to_condition(original_line),
                'severity': 'suggestion'
            })

        # Check for assignment in conditions (common bug)
        if 'if' in stripped_line and ' = ' in stripped_line and ' == ' not in stripped_line:
            self.warnings.append({
                'line': line_num,
                'message': "Possible assignment in condition - did you mean '=='?",
                'severity': 'warning'
            })

        # Check try-catch structure
        self._validate_try_catch_structure(stripped_line, line_num, original_line)

    def _validate_try_catch_structure(self, stripped_line, line_num, original_line):
        """Validate try-catch block structure"""
        # Check for try without catch
        if stripped_line.startswith('try') and 'catch' not in stripped_line:
            # Look ahead in context would be better, but for now we warn
            self.warnings.append({
                'line': line_num,
                'message': "Try block should be followed by catch block",
                'severity': 'warning'
            })

        # Check for malformed catch blocks
        if 'catch' in stripped_line:
            # Check for catch without error parameter
            if 'catch' in stripped_line and 'catch{' in stripped_line.replace(' ', ''):
                self.suggestions.append({
                    'line': line_num,
                    'message': "Catch should include error parameter: catch(error)",
                    'fix': original_line.replace('catch{', 'catch(error){'),
                    'severity': 'error'
                })

    def _fix_catch_syntax(self, line):
        """Fix catch syntax from 'catch error { }' to 'catch(error) { }'"""
        # Pattern: catch followed by identifier then {
        pattern = r'catch\s+(\w+)\s*\{'
        replacement = r'catch(\1) {'
        fixed_line = re.sub(pattern, replacement, line)
        
        if fixed_line != line:
            return fixed_line
        
        # Fallback: simple string replacement
        if 'catch ' in line and '{' in line:
            # Extract the error variable name
            parts = line.split('catch ', 1)[1].split('{', 1)
            if len(parts) == 2:
                error_var = parts[0].strip()
                rest = parts[1]
                return line.split('catch ', 1)[0] + f'catch({error_var}) {{' + rest
        
        return line

    def _fix_double_parentheses_catch(self, line):
        """Fix catch with double parentheses: catch((error)) to catch(error)"""
        # Pattern: catch((...))
        pattern = r'catch\(\s*\(\s*(\w+)\s*\)\s*\)'
        replacement = r'catch(\1)'
        fixed_line = re.sub(pattern, replacement, line)
        
        if fixed_line != line:
            return fixed_line
        
        # Fallback: simple string replacement
        if 'catch((' in line and '))' in line:
            start = line.find('catch((') + 7  # after 'catch(('
            end = line.find('))', start)
            if start > 0 and end > start:
                error_var = line[start:end].strip()
                return line.replace(f'catch(({error_var}))', f'catch({error_var})')
        
        return line

    def _fix_lambda_syntax(self, line):
        """Fix lambda syntax to universal style"""
        if 'lambda ' in line:
            # Simple case: lambda x: expr -> lambda(x) -> expr
            if ':' in line:
                return line.replace('lambda ', 'lambda(', 1).replace(':', ') ->', 1)
            # Case with arrow: lambda x -> expr -> lambda(x) -> expr
            elif '->' in line:
                return line.replace('lambda ', 'lambda(', 1).replace(' ->', ') ->', 1)
        return line

    def _add_parentheses_to_condition(self, line):
        """Add parentheses around condition"""
        if line.startswith('if '):
            return line.replace('if ', 'if (', 1) + ')'
        elif line.startswith('while '):
            return line.replace('while ', 'while (', 1) + ')'
        return line

    def auto_fix(self, code, desired_style="universal"):
        """Attempt to automatically fix syntax issues"""
        validation = self.validate_code(code, desired_style)

        if validation['is_valid']:
            return code, validation

        lines = code.split('\n')
        fixed_lines = lines.copy()
        applied_fixes = 0

        # Group suggestions by line to avoid conflicts
        line_suggestions = {}
        for suggestion in validation['suggestions']:
            if suggestion['severity'] in ['error', 'warning']:
                line_num = suggestion['line'] - 1
                if line_num not in line_suggestions:
                    line_suggestions[line_num] = []
                line_suggestions[line_num].append(suggestion)

        # Apply fixes line by line
        for line_num, suggestions in line_suggestions.items():
            if line_num < len(fixed_lines):
                current_line = fixed_lines[line_num]
                fixed_line = current_line
                
                # Apply fixes in order
                for suggestion in suggestions:
                    fixed_line = suggestion['fix']
                    applied_fixes += 1
                
                fixed_lines[line_num] = fixed_line

        fixed_code = '\n'.join(fixed_lines)

        # Re-validate after fixes
        final_validation = self.validate_code(fixed_code, desired_style)
        final_validation['applied_fixes'] = applied_fixes

        return fixed_code, final_validation

    def suggest_syntax_style(self, code):
        """Analyze code and suggest which syntax style it follows"""
        lines = code.split('\n')

        universal_indicators = 0
        tolerable_indicators = 0

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            # Universal indicators
            if any(stripped.startswith(kw + '(') for kw in ['if', 'while', 'debug']):
                universal_indicators += 1
            if 'lambda(' in stripped:
                universal_indicators += 1
            if stripped.endswith('{'):
                universal_indicators += 1
            if 'catch(' in stripped and 'catch((' not in stripped:
                universal_indicators += 1

            # Tolerable indicators
            if any(stripped.startswith(kw + ' ') and stripped.endswith(':') 
                   for kw in ['if', 'for each', 'while', 'action']):
                tolerable_indicators += 1
            if 'debug ' in stripped and not stripped.startswith('debug('):
                tolerable_indicators += 1
            if 'lambda ' in stripped and not 'lambda(' in stripped:
                tolerable_indicators += 1
            if 'catch ' in stripped and not 'catch(' in stripped:
                tolerable_indicators += 1
            if 'catch((' in stripped:
                tolerable_indicators += 1

        if universal_indicators > tolerable_indicators:
            return "universal"
        elif tolerable_indicators > universal_indicators:
            return "tolerable"
        else:
            return "mixed"

    def get_fix_summary(self, validation_result):
        """Get a summary of fixes that will be applied"""
        error_fixes = [s for s in validation_result['suggestions'] if s['severity'] == 'error']
        warning_fixes = [s for s in validation_result['suggestions'] if s['severity'] == 'warning']
        
        return {
            'total_fixes': len(validation_result['suggestions']),
            'error_fixes': len(error_fixes),
            'warning_fixes': len(warning_fixes),
            'will_fix_errors': len(error_fixes) > 0,
            'will_fix_warnings': len(warning_fixes) > 0
        }